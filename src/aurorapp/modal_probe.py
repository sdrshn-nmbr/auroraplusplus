import contextlib
import json
import os
import platform
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import modal

from aurorapp.artifacts import ContentAddressedArtifactStore
from aurorapp.canonical import canonical_bytes, canonical_sha256, file_sha256
from aurorapp.model_port import (
    CandidateSpeculativeServingResult,
    CapturedBatchOptimizerResult,
    CheckpointReferenceResult,
    CheckpointReloadResult,
    ParentDrafterRestoreResult,
    PhysicalModelPortResult,
    candidate_serving_from_probe_payload,
)
from aurorapp.models import ArtifactRef
from aurorapp.sglang_contract import SGLANG_SERVER_RANDOM_SEED, greedy_generation_request

TARGET_REPOSITORY = "poolside/Laguna-XS-2.1-INT4"
TARGET_REVISION = "4b7e28abdc0a8b121def816b89d631750bc53c92"
DRAFT_REPOSITORY = "poolside/Laguna-XS-2.1-DFlash-INT4"
DRAFT_REVISION = "630267be6a2ad870ca80fc3930bde87d6ad7bb24"
SGLANG_REVISION = "6a5a9eccaab0b98f5c3cfee170c7fbc87dfb98e5"
SPECFORGE_REVISION = "e6440f09a8574b35f894608559fd3d165971e488"
DFLASH_TARGET_LAYER_IDS = (1, 13, 25, 33, 39)
LAGUNA_HIDDEN_SIZE = 2048
CAPTURE_FEATURE_NAME = "target_aux_hidden_states"

app = modal.App("aurorapp-compatibility")

CUDA_IMAGE = "nvidia/cuda@sha256:6b6617592b94e7dcc6ffbe6d00720eed27bc6e3b4f06b26b93b4070c31f57391"
runtime_base_image = modal.Image.from_registry(CUDA_IMAGE, add_python="3.12").apt_install(
    "git", "pciutils", "libcurl4", "libibverbs1", "librdmacm1", "libnuma1"
)
base_image = runtime_base_image.add_local_python_source("aurorapp")

sglang_runtime_image = (
    runtime_base_image.add_local_file(
        "patches/sglang/6a5a9ec/spec-capture.patch",
        "/opt/aurorapp/spec-capture.patch",
        copy=True,
    )
    .env({"SGLANG_BUILD_RUST_EXTS": "none"})
    .run_commands(
        "git clone --filter=blob:none https://github.com/sgl-project/sglang.git /opt/sglang",
        f"git -C /opt/sglang checkout {SGLANG_REVISION}",
        "git -C /opt/sglang apply --check /opt/aurorapp/spec-capture.patch",
        "git -C /opt/sglang apply /opt/aurorapp/spec-capture.patch",
        "python -m pip install --upgrade pip uv",
        "uv pip install --system -e '/opt/sglang/python[all]'",
        "uv pip install --system mooncake-transfer-engine-cuda13==0.3.12.post1",
        (
            'python -c "import subprocess; '
            "output = subprocess.run(['ldd', "
            "'/usr/local/lib/python3.12/site-packages/mooncake/mooncake_master'], "
            "check=True, capture_output=True, text=True).stdout; print(output); "
            "missing = [line for line in output.splitlines() "
            "if 'not found' in line and 'libcuda.so.1' not in line]; "
            'assert not missing, missing"'
        ),
        "uv pip uninstall --system sgl-deep-gemm",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_HOME": "/usr/local/cuda",
            "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
        }
    )
)

sglang_image = sglang_runtime_image.add_local_python_source("aurorapp")

specforge_image = (
    sglang_runtime_image.add_local_file(
        "scripts/specforge_ingest_probe.py",
        "/opt/aurorapp/specforge_ingest_probe.py",
        copy=True,
    )
    .add_local_file(
        "scripts/specforge_model_port_probe.py",
        "/opt/aurorapp/specforge_model_port_probe.py",
        copy=True,
    )
    .add_local_file(
        "scripts/specforge_captured_optimizer_probe.py",
        "/opt/aurorapp/specforge_captured_optimizer_probe.py",
        copy=True,
    )
    .add_local_file(
        "patches/specforge/e6440f09/laguna-dflash-training.patch",
        "/opt/aurorapp/laguna-dflash-training.patch",
        copy=True,
    )
    .run_commands(
        "git clone --filter=blob:none https://github.com/sgl-project/SpecForge.git /opt/specforge",
        f"git -C /opt/specforge checkout {SPECFORGE_REVISION}",
        f'test "$(git -C /opt/specforge rev-parse HEAD)" = {SPECFORGE_REVISION}',
        "git -C /opt/specforge apply --check /opt/aurorapp/laguna-dflash-training.patch",
        "git -C /opt/specforge apply /opt/aurorapp/laguna-dflash-training.patch",
        "uv pip install --system --no-deps -e /opt/specforge",
        (
            'python -c "from specforge.inference.adapters.server_capture import '
            "SGLangServerCaptureAdapter; from specforge.runtime.data_plane."
            "mooncake_store import MooncakeFeatureStore; "
            'print(SGLangServerCaptureAdapter, MooncakeFeatureStore)"'
        ),
    )
    .add_local_python_source("aurorapp")
)

model_cache = modal.Volume.from_name("aurorapp-model-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    "aurorapp-compatibility-checkpoints", create_if_missing=True
)


def _run(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {' '.join(command)}\n{result.stderr}"
        )
    return result.stdout.strip()


def _hardware_identity() -> dict[str, Any]:
    query = (
        "uuid,name,memory.total,driver_version,pci.bus_id,power.limit,"
        "clocks.max.sm,clocks.max.memory"
    )
    gpu_rows = _run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    ).splitlines()
    return {
        "gpu_rows": gpu_rows,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "nvidia_smi_hash_input": "\n".join(gpu_rows),
    }


def _runtime_identity(repository_revision: str) -> dict[str, Any]:
    return {
        "aurorapp_repository_revision": repository_revision,
        "cuda_base_image": CUDA_IMAGE,
        "python": platform.python_version(),
        "packages": _run(["python", "-m", "pip", "freeze"]).splitlines(),
        "cuda_toolkit": _run(["nvcc", "--version"]),
        "sglang_revision": SGLANG_REVISION,
        "specforge_revision": SPECFORGE_REVISION,
        "build_environment": {
            "SGLANG_BUILD_RUST_EXTS": "none",
            "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "CUDA_HOME": "/usr/local/cuda",
        },
        "removed_packages": ["sgl-deep-gemm"],
    }


@app.function(
    image=base_image,
    gpu="H100!",
    timeout=300,
    single_use_containers=True,
    restrict_modal_access=True,
    name="exact-h100-identity",
)
def exact_h100_identity() -> dict[str, Any]:
    return _hardware_identity()


def _wait_for_server(port: int, process: subprocess.Popen[str], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"SGLang exited before health check with status {process.returncode}"
            )
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError("SGLang did not become healthy before the compatibility timeout")


def _generate(port: int) -> dict[str, Any]:
    payload = greedy_generation_request(
        "Write a Python function that adds two tensors.", max_new_tokens=32
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=canonical_bytes(payload),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as error:
        response_bytes = error.read()
        return {
            "request": payload,
            "response_status": error.code,
            "response_body": response_bytes.decode("utf-8", errors="replace"),
            "passed": False,
        }
    decoded = json.loads(response_bytes)
    if not isinstance(decoded, dict):
        return {
            "request": payload,
            "response_status": 200,
            "response_body": response_bytes.decode("utf-8", errors="replace"),
            "passed": False,
            "error": "SGLang generation response was not a JSON object",
        }
    return {
        "request": payload,
        "response_status": 200,
        "response_body": decoded,
        "passed": True,
    }


def _capture_generate(port: int) -> dict[str, Any]:
    payload = {
        "text": "Write a Python function that adds two tensors.",
        "extra_key": "aurorapp-laguna-capture-probe",
        "sampling_params": {"temperature": 0, "max_new_tokens": 0},
        "spec_capture": {
            "store_id": "aurorapp-compatibility",
            "sample_id": "laguna-dflash-capture-1",
            "gen": 1,
            "replace": True,
            "features": {"aux": CAPTURE_FEATURE_NAME},
        },
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=canonical_bytes(payload),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as error:
        return {
            "request": payload,
            "response_status": error.code,
            "response_body": error.read().decode("utf-8", errors="replace"),
            "passed": False,
        }
    decoded = json.loads(response_bytes)
    if not isinstance(decoded, dict):
        return {
            "request": payload,
            "response_status": 200,
            "response_body": response_bytes.decode("utf-8", errors="replace"),
            "passed": False,
            "error": "SGLang capture response was not a JSON object",
        }
    capture = (decoded.get("meta_info") or {}).get("spec_capture")
    if isinstance(capture, list) and len(capture) == 1:
        capture = capture[0]
    try:
        token_count, feature_width = validate_dflash_capture_result(capture)
    except (TypeError, ValueError) as error:
        return {
            "request": payload,
            "response_status": 200,
            "response_body": decoded,
            "passed": False,
            "error": str(error),
        }
    return {
        "request": payload,
        "response_status": 200,
        "response_body": decoded,
        "capture_result": capture,
        "token_count": token_count,
        "feature_width": feature_width,
        "passed": True,
    }


def validate_dflash_capture_result(result: object) -> tuple[int, int]:
    if not isinstance(result, dict):
        raise ValueError("capture response has no result object")
    if result.get("error"):
        raise ValueError(f"capture sink failed: {result['error']}")
    if tuple(result.get("aux_layer_ids") or ()) != DFLASH_TARGET_LAYER_IDS:
        raise ValueError("capture used the wrong Laguna DFlash target layers")
    features = result.get("features")
    if not isinstance(features, dict):
        raise ValueError("capture result has no feature manifest")
    feature = features.get(CAPTURE_FEATURE_NAME)
    if not isinstance(feature, dict):
        raise ValueError(f"capture result has no {CAPTURE_FEATURE_NAME} feature")
    shape = feature.get("shape")
    expected_width = len(DFLASH_TARGET_LAYER_IDS) * LAGUNA_HIDDEN_SIZE
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or shape[0] != 1
        or shape[1] < 1
        or shape[2] != expected_width
    ):
        raise ValueError(
            f"capture feature shape {shape!r} does not match [1, tokens, {expected_width}]"
        )
    if feature.get("dtype") != "bfloat16":
        raise ValueError("capture feature dtype is not bfloat16")
    return int(shape[1]), int(shape[2])


def validate_specforge_ingest_result(
    result: object,
    expected_input_ids: list[int],
    expected_loss_mask: list[int],
) -> tuple[int, int]:
    if not isinstance(result, dict):
        raise ValueError("SpecForge ingest returned no result object")
    if result.get("strategy") != "dflash":
        raise ValueError("SpecForge ingest did not produce a DFlash sample")
    if not result.get("sample_id"):
        raise ValueError("SpecForge ingest returned no sample identity")
    materialized = result.get("materialized")
    if not isinstance(materialized, dict):
        raise ValueError("SpecForge did not materialize the captured tensors")

    def tensor(name: str) -> dict[str, Any]:
        value = materialized.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"SpecForge batch is missing materialized {name}")
        return value

    input_ids = tensor("input_ids")
    loss_mask = tensor("loss_mask")
    hidden_states = tensor("hidden_states")
    token_count = len(expected_input_ids)
    if input_ids.get("shape") != [1, token_count] or input_ids.get("dtype") != "int64":
        raise ValueError("SpecForge input IDs have the wrong tensor contract")
    if input_ids.get("values") != expected_input_ids:
        raise ValueError("SpecForge input IDs differ from the captured request")
    if loss_mask.get("shape") != [1, token_count] or loss_mask.get("dtype") != "int64":
        raise ValueError("SpecForge loss mask has the wrong tensor contract")
    if loss_mask.get("values") != expected_loss_mask:
        raise ValueError("SpecForge loss mask differs from the captured request")
    if hidden_states.get("shape") != [1, token_count, 10240]:
        raise ValueError("SpecForge hidden states have the wrong Laguna DFlash shape")
    if hidden_states.get("dtype") != "bfloat16":
        raise ValueError("SpecForge hidden states are not bfloat16")
    if hidden_states.get("finite") is not True:
        raise ValueError("SpecForge hidden states contain non-finite values")
    absolute_sum = hidden_states.get("absolute_sum")
    if not isinstance(absolute_sum, int | float) or absolute_sum <= 0:
        raise ValueError("SpecForge hidden states are empty or all zero")
    release_drain = result.get("release_drain")
    if (
        result.get("released") is not True
        or not isinstance(release_drain, dict)
        or release_drain.get("release_pending") != 0
    ):
        raise ValueError("SpecForge did not release the materialized batch")
    return token_count, 10240


def _specforge_ingest(port: int) -> dict[str, Any]:
    command = [
        "python",
        "/opt/aurorapp/specforge_ingest_probe.py",
        "--base-url",
        f"http://127.0.0.1:{port}",
    ]
    execution = _run_until_terminal_record(
        command,
        marker="AURORAPP_RESULT=",
        timeout=300,
    )
    if execution["timed_out"]:
        return {
            "passed": False,
            "command": command,
            "returncode": execution["returncode"],
            "stdout": execution["output"],
            "stderr": "",
            "error": "SpecForge ingest subprocess timed out after 300 seconds",
        }
    result_lines = execution["terminal_records"]
    accepted_exit = (
        execution["returncode"] == 0 or execution["terminated_after_terminal_record"] is True
    )
    if not accepted_exit or len(result_lines) != 1:
        return {
            "passed": False,
            "command": command,
            "returncode": execution["returncode"],
            "stdout": execution["output"],
            "stderr": "",
            "error": "SpecForge ingest subprocess did not return one result",
        }
    try:
        result = json.loads(result_lines[0])
        token_count, feature_width = validate_specforge_ingest_result(
            result,
            [1, 2, 3, 4],
            [0, 0, 1, 1],
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return {
            "passed": False,
            "command": command,
            "returncode": execution["returncode"],
            "stdout": execution["output"],
            "stderr": "",
            "error": str(error),
        }
    return {
        "passed": True,
        "command": command,
        "returncode": execution["returncode"],
        "stdout": execution["output"],
        "stderr": "",
        "terminated_after_terminal_record": execution["terminated_after_terminal_record"],
        "result": result,
        "token_count": token_count,
        "feature_width": feature_width,
    }


def _specforge_ingest_after_prewarm(port: int) -> dict[str, Any]:
    prewarm = _capture_generate(port)
    if prewarm.get("passed") is not True:
        return {
            "passed": False,
            "prewarm": prewarm,
            "error": "Mooncake capture sink prewarm failed",
        }
    ingest = _specforge_ingest(port)
    return {**ingest, "prewarm": prewarm}


def _model_port_optimizer() -> dict[str, Any]:
    command = [
        "python",
        "/opt/aurorapp/specforge_model_port_probe.py",
        "--repository",
        DRAFT_REPOSITORY,
        "--revision",
        DRAFT_REVISION,
    ]
    execution = _run_until_terminal_record(
        command,
        marker="AURORAPP_RESULT=",
        timeout=900,
    )
    records = execution["terminal_records"]
    if execution["timed_out"] or len(records) != 1:
        return {
            "passed": False,
            "command": command,
            "execution": execution,
            "error": "model port probe did not return one terminal record",
        }
    try:
        result = PhysicalModelPortResult.model_validate_json(records[0])
    except (ValueError, TypeError) as error:
        return {
            "passed": False,
            "command": command,
            "execution": execution,
            "error": str(error),
        }
    return {
        "passed": result.passed,
        "command": command,
        "execution": execution,
        "result": result.model_dump(mode="json"),
    }


def _captured_batch_optimizer(port: int) -> dict[str, Any]:
    command = [
        "python",
        "/opt/aurorapp/specforge_captured_optimizer_probe.py",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--target-repository",
        TARGET_REPOSITORY,
        "--target-revision",
        TARGET_REVISION,
        "--draft-repository",
        DRAFT_REPOSITORY,
        "--draft-revision",
        DRAFT_REVISION,
        "--checkpoint-root",
        "/checkpoints",
    ]
    execution = _run_until_terminal_record(
        command,
        marker="AURORAPP_RESULT=",
        timeout=1200,
    )
    records = execution["terminal_records"]
    if execution["timed_out"] or len(records) != 1:
        return {
            "passed": False,
            "command": command,
            "execution": execution,
            "error": "captured optimizer probe did not return one terminal record",
        }
    try:
        result = CapturedBatchOptimizerResult.model_validate_json(records[0])
    except (ValueError, TypeError) as error:
        return {
            "passed": False,
            "command": command,
            "execution": execution,
            "error": str(error),
        }
    if result.passed:
        checkpoint_volume.commit()
    return {
        "passed": result.passed,
        "command": command,
        "execution": execution,
        "result": result.model_dump(mode="json"),
        "volume_committed": result.passed,
    }


def _checkpoint_reload_diagnostic(checkpoint_path: str) -> dict[str, Any]:
    reference_path = "/tmp/aurorapp-checkpoint-reference.pt"
    reference_command = [
        "python",
        "/opt/aurorapp/specforge_captured_optimizer_probe.py",
        "--reload-checkpoint",
        checkpoint_path,
        "--write-reference",
        reference_path,
    ]
    reference_execution = _run_until_terminal_record(
        reference_command,
        marker="AURORAPP_REFERENCE_RESULT=",
        timeout=600,
    )
    reference_records = reference_execution["terminal_records"]
    if reference_execution["timed_out"] or len(reference_records) != 1:
        return {
            "passed": False,
            "reference_command": reference_command,
            "reference_execution": reference_execution,
            "error": "checkpoint reference process did not return one terminal record",
        }
    try:
        reference = CheckpointReferenceResult.model_validate_json(reference_records[0])
    except (ValueError, TypeError) as error:
        return {
            "passed": False,
            "reference_command": reference_command,
            "reference_execution": reference_execution,
            "error": str(error),
        }
    reload_command = [
        "python",
        "/opt/aurorapp/specforge_captured_optimizer_probe.py",
        "--reload-checkpoint",
        checkpoint_path,
        "--reload-reference",
        reference_path,
    ]
    reload_execution = _run_until_terminal_record(
        reload_command,
        marker="AURORAPP_RELOAD_RESULT=",
        timeout=600,
    )
    reload_records = reload_execution["terminal_records"]
    if reload_execution["timed_out"] or len(reload_records) != 1:
        return {
            "passed": False,
            "reference": reference.model_dump(mode="json"),
            "reference_execution": reference_execution,
            "reload_command": reload_command,
            "reload_execution": reload_execution,
            "error": "checkpoint reload process did not return one terminal record",
        }
    try:
        reload = CheckpointReloadResult.model_validate_json(reload_records[0])
    except (ValueError, TypeError) as error:
        return {
            "passed": False,
            "reference": reference.model_dump(mode="json"),
            "reference_execution": reference_execution,
            "reload_command": reload_command,
            "reload_execution": reload_execution,
            "error": str(error),
        }
    return {
        "passed": reload.passed,
        "checkpoint_path": checkpoint_path,
        "reference": reference.model_dump(mode="json"),
        "reference_execution": reference_execution,
        "reload": reload.model_dump(mode="json"),
        "reload_execution": reload_execution,
    }


def _port_is_closed(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(1)
        return client.connect_ex(("127.0.0.1", port)) != 0


def _gpu_processes() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return result.stdout.splitlines()


def _stop_process_group(process: subprocess.Popen[str], port: int) -> dict[str, Any]:
    process_group = os.getpgid(process.pid) if process.poll() is None else process.pid
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group, signal.SIGTERM)
    if process.poll() is None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=30)
    remaining = _process_group_members(process_group)
    if remaining:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            remaining = _process_group_members(process_group)
            if not remaining:
                break
            time.sleep(0.1)
    if process.poll() is None:
        process.wait(timeout=10)
    port_deadline = time.monotonic() + 5
    while time.monotonic() < port_deadline and not _port_is_closed(port):
        time.sleep(0.1)
    return {
        "process_group": process_group,
        "remaining_processes": remaining,
        "gpu_processes": _gpu_processes(),
        "port_closed": _port_is_closed(port),
        "parent_returncode": process.returncode,
    }


def _process_group_members(process_group: int) -> list[str]:
    return subprocess.run(
        ["pgrep", "-g", str(process_group)],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def _start_logged_process(command: list[str]) -> tuple[subprocess.Popen[str], Path]:
    file_descriptor, raw_path = tempfile.mkstemp(prefix="aurorapp-service-", suffix=".log")
    log_path = Path(raw_path)
    log_stream = os.fdopen(file_descriptor, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except BaseException:
        log_stream.close()
        log_path.unlink(missing_ok=True)
        raise
    log_stream.close()
    return process, log_path


def _read_process_log(log_path: Path) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    finally:
        log_path.unlink(missing_ok=True)


def _run_until_terminal_record(
    command: list[str],
    *,
    marker: str,
    timeout: int,
) -> dict[str, Any]:
    process, log_path = _start_logged_process(command)
    deadline = time.monotonic() + timeout
    terminal_records: list[str] = []
    timed_out = False
    while time.monotonic() < deadline:
        output = log_path.read_text(encoding="utf-8", errors="replace")
        terminal_records = [
            line.removeprefix(marker).rstrip("\r\n")
            for line in output.splitlines(keepends=True)
            if line.startswith(marker) and line.endswith(("\n", "\r"))
        ]
        if terminal_records or process.poll() is not None:
            break
        time.sleep(0.05)
    else:
        timed_out = True

    terminated_after_terminal_record = bool(terminal_records and process.poll() is None)
    if process.poll() is None:
        process_group = os.getpgid(process.pid)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process_group, signal.SIGKILL)
            process.wait(timeout=10)

    output = _read_process_log(log_path)
    terminal_records = [
        line.removeprefix(marker).rstrip("\r\n")
        for line in output.splitlines(keepends=True)
        if line.startswith(marker) and line.endswith(("\n", "\r"))
    ]
    return {
        "output": output,
        "returncode": process.returncode,
        "terminal_records": terminal_records,
        "terminated_after_terminal_record": terminated_after_terminal_record,
        "timed_out": timed_out,
    }


def _server_probe(speculative: bool, repository_revision: str) -> dict[str, Any]:
    port = 31000 if speculative else 30000
    command = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        TARGET_REPOSITORY,
        "--revision",
        TARGET_REVISION,
        "--trust-remote-code",
        "--reasoning-parser",
        "poolside_v1",
        "--tool-call-parser",
        "poolside_v1",
        "--tp",
        "1",
        "--attention-backend",
        "fa3",
        "--page-size",
        "1",
        "--mem-fraction-static",
        "0.7",
        "--random-seed",
        str(SGLANG_SERVER_RANDOM_SEED),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if speculative:
        command.extend(
            [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                DRAFT_REPOSITORY,
                "--speculative-draft-model-revision",
                DRAFT_REVISION,
            ]
        )
    process, log_path = _start_logged_process(command)
    generation: dict[str, Any] = {"passed": False, "error": "generation did not run"}
    server_healthy = False
    error: dict[str, str] | None = None
    try:
        _wait_for_server(port, process, timeout=900)
        server_healthy = True
        generation = _generate(port)
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    finally:
        cleanup = _stop_process_group(process, port)
        stdout = _read_process_log(log_path)
    cleanup_passed = (
        not cleanup["remaining_processes"]
        and not cleanup["gpu_processes"]
        and cleanup["port_closed"]
    )
    return {
        "status": (
            "passed"
            if server_healthy and generation.get("passed") is True and cleanup_passed
            else "failed"
        ),
        "hardware": _hardware_identity(),
        "runtime": _runtime_identity(repository_revision),
        "server_healthy": server_healthy,
        "generation": generation,
        "speculative": speculative,
        "launch_command": command,
        "error": error,
        "cleanup": cleanup,
        "cleanup_passed": cleanup_passed,
        "sglang_log": stdout,
    }


def _candidate_serving_arm(
    *,
    port: int,
    speculative_draft_path: str | None,
) -> dict[str, Any]:
    command = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        TARGET_REPOSITORY,
        "--revision",
        TARGET_REVISION,
        "--trust-remote-code",
        "--reasoning-parser",
        "poolside_v1",
        "--tool-call-parser",
        "poolside_v1",
        "--tp",
        "1",
        "--attention-backend",
        "fa3",
        "--page-size",
        "1",
        "--mem-fraction-static",
        "0.7",
        "--random-seed",
        str(SGLANG_SERVER_RANDOM_SEED),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if speculative_draft_path is not None:
        command.extend(
            [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                speculative_draft_path,
            ]
        )
    process, log_path = _start_logged_process(command)
    generation: dict[str, Any] = {"passed": False, "error": "generation did not run"}
    server_healthy = False
    error: dict[str, str] | None = None
    try:
        _wait_for_server(port, process, timeout=900)
        server_healthy = True
        generation = _generate(port)
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    finally:
        cleanup = _stop_process_group(process, port)
        server_log = _read_process_log(log_path)
    cleanup_passed = (
        not cleanup["remaining_processes"]
        and not cleanup["gpu_processes"]
        and cleanup["port_closed"]
    )
    return {
        "passed": server_healthy and generation.get("passed") is True and cleanup_passed,
        "server_healthy": server_healthy,
        "generation": generation,
        "launch_command": command,
        "error": error,
        "cleanup": cleanup,
        "cleanup_passed": cleanup_passed,
        "sglang_log": server_log,
    }


def _candidate_serving_result(
    *,
    target_arm: dict[str, Any],
    candidate_arm: dict[str, Any],
    checkpoint_path: str,
    manifest_hash: str,
    weights_hash: str,
) -> CandidateSpeculativeServingResult:
    target_generation = target_arm["generation"]
    candidate_generation = candidate_arm["generation"]
    target_request = target_generation["request"]
    candidate_request = candidate_generation["request"]
    if target_request != candidate_request:
        raise ValueError("target-only and candidate requests differ")
    target_response = target_generation["response_body"]
    candidate_response = candidate_generation["response_body"]
    target_meta = target_response["meta_info"]
    candidate_meta = candidate_response["meta_info"]
    telemetry = {
        "proposed_drafts": candidate_meta.get(
            "spec_proposed_drafts", candidate_meta.get("spec_num_proposed_drafts", 0)
        ),
        "accepted_drafts": candidate_meta.get(
            "spec_accepted_drafts", candidate_meta.get("spec_num_correct_drafts", 0)
        ),
        "verify_count": candidate_meta.get("spec_verify_ct", 0),
        "accept_histogram": candidate_meta.get("spec_accept_histogram", []),
    }
    return CandidateSpeculativeServingResult.model_validate(
        {
            "target_repository": TARGET_REPOSITORY,
            "target_revision": TARGET_REVISION,
            "parent_draft_repository": DRAFT_REPOSITORY,
            "parent_draft_revision": DRAFT_REVISION,
            "candidate_checkpoint_path": checkpoint_path,
            "candidate_manifest_hash": manifest_hash,
            "candidate_weights_hash": weights_hash,
            "request_hash": canonical_sha256(target_request),
            "target_output_ids": target_response["output_ids"],
            "candidate_output_ids": candidate_response["output_ids"],
            "target_text": target_response["text"],
            "candidate_text": candidate_response["text"],
            "target_finish_reason": target_meta["finish_reason"],
            "candidate_finish_reason": candidate_meta["finish_reason"],
            "speculative_telemetry": telemetry,
            "target_server_healthy": target_arm["server_healthy"],
            "candidate_server_healthy": candidate_arm["server_healthy"],
            "draft_checkpoint_loaded": candidate_arm["server_healthy"]
            and telemetry["proposed_drafts"] > 0,
            "target_cleanup_passed": target_arm["cleanup_passed"],
            "candidate_cleanup_passed": candidate_arm["cleanup_passed"],
        }
    )


def _speculative_telemetry(response: dict[str, Any]) -> dict[str, Any]:
    meta = response["meta_info"]
    return {
        "proposed_drafts": meta.get(
            "spec_proposed_drafts", meta.get("spec_num_proposed_drafts", 0)
        ),
        "accepted_drafts": meta.get(
            "spec_accepted_drafts", meta.get("spec_num_correct_drafts", 0)
        ),
        "verify_count": meta.get("spec_verify_ct", 0),
        "accept_histogram": meta.get("spec_accept_histogram", []),
    }


def _parent_restore_probe(
    repository_revision: str,
    candidate_serving_evidence_hash: str,
    candidate_manifest_hash: str,
    request_hash: str,
    expected_output_ids: list[int],
    expected_text: str,
    expected_finish_reason: dict[str, Any],
) -> dict[str, Any]:
    parent = _server_probe(speculative=True, repository_revision=repository_revision)
    try:
        generation = parent["generation"]
        request = generation["request"]
        response = generation["response_body"]
        if canonical_sha256(request) != request_hash:
            raise ValueError("parent restore request differs from candidate serving request")
        result = ParentDrafterRestoreResult.model_validate(
            {
                "candidate_serving_evidence_hash": candidate_serving_evidence_hash,
                "candidate_manifest_hash": candidate_manifest_hash,
                "parent_draft_repository": DRAFT_REPOSITORY,
                "parent_draft_revision": DRAFT_REVISION,
                "request_hash": request_hash,
                "expected_output_ids": expected_output_ids,
                "parent_output_ids": response["output_ids"],
                "expected_text": expected_text,
                "parent_text": response["text"],
                "expected_finish_reason": expected_finish_reason,
                "parent_finish_reason": response["meta_info"]["finish_reason"],
                "speculative_telemetry": _speculative_telemetry(response),
                "server_healthy": parent["server_healthy"],
                "cleanup_passed": parent["cleanup_passed"],
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        return {
            "status": "failed",
            "hardware": _hardware_identity(),
            "runtime": _runtime_identity(repository_revision),
            "parent": parent,
            "cleanup_passed": parent.get("cleanup_passed") is True,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    return {
        "status": "passed" if result.passed else "failed",
        "hardware": _hardware_identity(),
        "runtime": _runtime_identity(repository_revision),
        "result": result.model_dump(mode="json"),
        "parent": parent,
        "cleanup_passed": parent["cleanup_passed"],
    }


def _candidate_speculative_serving_probe(
    repository_revision: str,
    checkpoint_path: str,
    manifest_hash: str,
    weights_hash: str,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_relative_to("/checkpoints/objects"):
        raise ValueError("candidate checkpoint must be an immutable checkpoint object")
    observed_manifest_hash = file_sha256(checkpoint / "manifest.json")
    observed_weights_hash = file_sha256(checkpoint / "model.safetensors")
    if observed_manifest_hash != manifest_hash:
        raise ValueError("candidate manifest hash differs from the requested checkpoint")
    if observed_weights_hash != weights_hash:
        raise ValueError("candidate weights hash differs from the requested checkpoint")

    target_arm = _candidate_serving_arm(port=35000, speculative_draft_path=None)
    candidate_arm = _candidate_serving_arm(
        port=36000,
        speculative_draft_path=checkpoint_path,
    )
    try:
        result = _candidate_serving_result(
            target_arm=target_arm,
            candidate_arm=candidate_arm,
            checkpoint_path=checkpoint_path,
            manifest_hash=manifest_hash,
            weights_hash=weights_hash,
        )
    except (KeyError, TypeError, ValueError) as error:
        return {
            "status": "failed",
            "hardware": _hardware_identity(),
            "runtime": _runtime_identity(repository_revision),
            "checkpoint": {
                "path": checkpoint_path,
                "manifest_hash": observed_manifest_hash,
                "weights_hash": observed_weights_hash,
            },
            "target_arm": target_arm,
            "candidate_arm": candidate_arm,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    return {
        "status": "passed" if result.passed else "failed",
        "hardware": _hardware_identity(),
        "runtime": _runtime_identity(repository_revision),
        "checkpoint": {
            "path": checkpoint_path,
            "manifest_hash": observed_manifest_hash,
            "weights_hash": observed_weights_hash,
        },
        "result": result.model_dump(mode="json"),
        "target_arm": target_arm,
        "candidate_arm": candidate_arm,
        "cleanup_passed": (
            target_arm["cleanup_passed"] and candidate_arm["cleanup_passed"]
        ),
    }


def _capture_server_probe(
    repository_revision: str,
    *,
    specforge_ingest: bool = False,
    captured_optimizer: bool = False,
) -> dict[str, Any]:
    if specforge_ingest and captured_optimizer:
        raise ValueError("capture probe accepts one SpecForge workload")
    port = 34000 if captured_optimizer else (33000 if specforge_ingest else 32000)
    os.environ.update(
        {
            "MOONCAKE_MASTER_SERVER_ADDR": "127.0.0.1:50051",
            "MOONCAKE_METADATA_SERVER": "http://127.0.0.1:8080/metadata",
            "MOONCAKE_LOCAL_HOSTNAME": "127.0.0.1",
            "MOONCAKE_PROTOCOL": "tcp",
        }
    )
    master, master_log_path = _start_logged_process(
        ["mooncake_master", "--enable-http-metadata-server=true"]
    )
    command = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        TARGET_REPOSITORY,
        "--revision",
        TARGET_REVISION,
        "--trust-remote-code",
        "--tp",
        "1",
        "--attention-backend",
        "fa3",
        "--page-size",
        "1",
        "--mem-fraction-static",
        "0.7",
        "--random-seed",
        str(SGLANG_SERVER_RANDOM_SEED),
        "--chunked-prefill-size",
        "-1",
        "--enable-spec-capture",
        "--spec-capture-method",
        "dflash",
        "--spec-capture-aux-layer-ids",
        *[str(layer_id) for layer_id in DFLASH_TARGET_LAYER_IDS],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    process, server_log_path = _start_logged_process(command)
    workload: dict[str, Any] = {"passed": False, "error": "capture did not run"}
    server_healthy = False
    error: dict[str, str] | None = None
    try:
        _wait_for_server(port, process, timeout=900)
        server_healthy = True
        if captured_optimizer:
            prewarm = _capture_generate(port)
            workload = (
                {**_captured_batch_optimizer(port), "prewarm": prewarm}
                if prewarm.get("passed") is True
                else {
                    "passed": False,
                    "prewarm": prewarm,
                    "error": "Mooncake capture sink prewarm failed",
                }
            )
        elif specforge_ingest:
            workload = _specforge_ingest_after_prewarm(port)
        else:
            workload = _capture_generate(port)
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    finally:
        server_cleanup = _stop_process_group(process, port)
        server_log = _read_process_log(server_log_path)
        master_cleanup = _stop_process_group(master, 50051)
        master_log = _read_process_log(master_log_path)
    cleanup_passed = (
        not server_cleanup["remaining_processes"]
        and not master_cleanup["remaining_processes"]
        and not master_cleanup["gpu_processes"]
        and server_cleanup["port_closed"]
        and master_cleanup["port_closed"]
    )
    return {
        "status": (
            "passed"
            if server_healthy and workload.get("passed") is True and cleanup_passed
            else "failed"
        ),
        "hardware": _hardware_identity(),
        "runtime": _runtime_identity(repository_revision),
        "server_healthy": server_healthy,
        (
            "captured_optimizer"
            if captured_optimizer
            else ("ingest" if specforge_ingest else "capture")
        ): workload,
        "launch_command": command,
        "error": error,
        "cleanup": {"server": server_cleanup, "mooncake": master_cleanup},
        "cleanup_passed": cleanup_passed,
        "sglang_log": server_log,
        "mooncake_log": master_log,
    }


@app.function(
    image=sglang_image,
    gpu="H100!",
    timeout=1800,
    volumes={"/root/.cache/huggingface": model_cache},
    single_use_containers=True,
    restrict_modal_access=True,
    name="target-only-probe",
)
def target_only_probe(repository_revision: str) -> dict[str, Any]:
    return _server_probe(speculative=False, repository_revision=repository_revision)


@app.function(
    image=sglang_image,
    gpu="H100!",
    timeout=1800,
    volumes={"/root/.cache/huggingface": model_cache},
    single_use_containers=True,
    restrict_modal_access=True,
    name="official-dflash-probe",
)
def official_dflash_probe(repository_revision: str) -> dict[str, Any]:
    return _server_probe(speculative=True, repository_revision=repository_revision)


@app.function(
    image=sglang_image,
    gpu="H100!",
    timeout=2400,
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/checkpoints": checkpoint_volume,
    },
    single_use_containers=True,
    restrict_modal_access=True,
    name="candidate-dflash-serving-probe",
)
def candidate_dflash_serving_probe(
    repository_revision: str,
    checkpoint_path: str,
    manifest_hash: str,
    weights_hash: str,
) -> dict[str, Any]:
    return _candidate_speculative_serving_probe(
        repository_revision,
        checkpoint_path,
        manifest_hash,
        weights_hash,
    )


@app.function(
    image=sglang_image,
    gpu="H100!",
    timeout=1800,
    volumes={"/root/.cache/huggingface": model_cache},
    single_use_containers=True,
    restrict_modal_access=True,
    name="parent-dflash-restore-probe",
)
def parent_dflash_restore_probe(
    repository_revision: str,
    candidate_serving_evidence_hash: str,
    candidate_manifest_hash: str,
    request_hash: str,
    expected_output_ids: list[int],
    expected_text: str,
    expected_finish_reason: dict[str, Any],
) -> dict[str, Any]:
    return _parent_restore_probe(
        repository_revision,
        candidate_serving_evidence_hash,
        candidate_manifest_hash,
        request_hash,
        expected_output_ids,
        expected_text,
        expected_finish_reason,
    )


@app.function(
    image=sglang_image,
    gpu="H100!",
    timeout=1800,
    volumes={"/root/.cache/huggingface": model_cache},
    single_use_containers=True,
    restrict_modal_access=True,
    name="laguna-dflash-capture-probe",
)
def laguna_dflash_capture_probe(repository_revision: str) -> dict[str, Any]:
    return _capture_server_probe(repository_revision)


@app.function(
    image=specforge_image,
    gpu="H100!",
    timeout=1800,
    volumes={"/root/.cache/huggingface": model_cache},
    single_use_containers=True,
    restrict_modal_access=True,
    name="specforge-batch-ingest-probe",
)
def specforge_batch_ingest_probe(repository_revision: str) -> dict[str, Any]:
    return _capture_server_probe(repository_revision, specforge_ingest=True)


@app.function(
    image=specforge_image,
    gpu="H100!",
    timeout=1800,
    volumes={"/root/.cache/huggingface": model_cache},
    single_use_containers=True,
    restrict_modal_access=True,
    name="specforge-laguna-model-port-probe",
)
def specforge_laguna_model_port_probe(repository_revision: str) -> dict[str, Any]:
    optimizer = _model_port_optimizer()
    gpu_processes = _gpu_processes()
    passed = optimizer.get("passed") is True and not gpu_processes
    return {
        "status": "passed" if passed else "failed",
        "hardware": _hardware_identity(),
        "runtime": _runtime_identity(repository_revision),
        "optimizer": optimizer,
        "cleanup": {"gpu_processes": gpu_processes},
        "cleanup_passed": not gpu_processes,
    }


@app.function(
    image=specforge_image,
    gpu="H100!",
    timeout=2400,
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/checkpoints": checkpoint_volume,
    },
    single_use_containers=True,
    restrict_modal_access=True,
    name="specforge-laguna-captured-optimizer-probe",
)
def specforge_laguna_captured_optimizer_probe(
    repository_revision: str,
) -> dict[str, Any]:
    return _capture_server_probe(repository_revision, captured_optimizer=True)


@app.function(
    image=specforge_image,
    gpu="H100!",
    timeout=1200,
    volumes={"/checkpoints": checkpoint_volume},
    single_use_containers=True,
    restrict_modal_access=True,
    name="specforge-laguna-reload-diagnostic",
)
def specforge_laguna_reload_diagnostic(
    repository_revision: str,
    checkpoint_path: str,
) -> dict[str, Any]:
    diagnostic = _checkpoint_reload_diagnostic(checkpoint_path)
    gpu_processes = _gpu_processes()
    passed = diagnostic.get("passed") is True and not gpu_processes
    return {
        "status": "passed" if passed else "failed",
        "hardware": _hardware_identity(),
        "runtime": _runtime_identity(repository_revision),
        "diagnostic": diagnostic,
        "cleanup": {"gpu_processes": gpu_processes},
        "cleanup_passed": not gpu_processes,
    }


@app.local_entrypoint()
def main(
    probe: str = "identity",
    output: str = "",
    allow_dirty: bool = False,
    checkpoint_path: str = "",
    checkpoint_manifest_hash: str = "",
    checkpoint_weights_hash: str = "",
    candidate_serving_evidence: str = "",
) -> None:
    revision = _local_repository_revision(allow_dirty)
    try:
        if probe == "identity":
            result = exact_h100_identity.remote()
        elif probe == "target":
            result = target_only_probe.remote(revision)
        elif probe == "dflash":
            result = official_dflash_probe.remote(revision)
        elif probe == "capture":
            result = laguna_dflash_capture_probe.remote(revision)
        elif probe == "ingest":
            result = specforge_batch_ingest_probe.remote(revision)
        elif probe == "model-port":
            result = specforge_laguna_model_port_probe.remote(revision)
        elif probe == "captured-optimizer":
            result = specforge_laguna_captured_optimizer_probe.remote(revision)
        elif probe == "reload-diagnostic":
            if not checkpoint_path:
                raise ValueError("reload-diagnostic requires --checkpoint-path")
            result = specforge_laguna_reload_diagnostic.remote(
                revision,
                checkpoint_path,
            )
        elif probe == "candidate-serving":
            if not checkpoint_path:
                raise ValueError("candidate-serving requires --checkpoint-path")
            if not checkpoint_manifest_hash or not checkpoint_weights_hash:
                raise ValueError(
                    "candidate-serving requires checkpoint manifest and weights hashes"
                )
            result = candidate_dflash_serving_probe.remote(
                revision,
                checkpoint_path,
                checkpoint_manifest_hash,
                checkpoint_weights_hash,
            )
        elif probe == "parent-restore":
            if not candidate_serving_evidence:
                raise ValueError("parent-restore requires --candidate-serving-evidence")
            candidate_record = json.loads(
                Path(candidate_serving_evidence).read_text(encoding="utf-8")
            )
            candidate = candidate_serving_from_probe_payload(
                candidate_record["result"]["result"]
            )
            if candidate_record.get("status") != "passed" or not candidate.passed:
                raise ValueError("parent-restore requires passing candidate evidence")
            artifact = ArtifactRef.model_validate(candidate_record["artifact"])
            result = parent_dflash_restore_probe.remote(
                revision,
                artifact.content_hash,
                candidate.candidate_manifest_hash,
                candidate.request_hash,
                list(candidate.candidate_output_ids),
                candidate.candidate_text,
                candidate.candidate_finish_reason.model_dump(mode="json"),
            )
        else:
            raise ValueError(
                "probe must be identity, target, dflash, capture, ingest, "
                "model-port, captured-optimizer, reload-diagnostic, candidate-serving, "
                "or parent-restore"
            )
        remote_status = result.get("status") if isinstance(result, dict) else None
        record: dict[str, Any] = {
            "probe": probe,
            "status": remote_status or "passed",
            "result": result,
            "repository_revision": revision,
            "modal_image_id": (
                (
                    specforge_image.object_id
                    if probe
                    in {
                        "ingest",
                        "model-port",
                        "captured-optimizer",
                        "reload-diagnostic",
                        "candidate-serving",
                        "parent-restore",
                    }
                    else sglang_image.object_id
                )
                if probe != "identity"
                else base_image.object_id
            ),
        }
    except Exception as error:
        record = {
            "probe": probe,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "repository_revision": revision,
        }
        artifact_store = ContentAddressedArtifactStore(Path("artifacts/compatibility/store"))
        artifact = _publish_record(artifact_store, probe, record)
        record["artifact"] = artifact.model_dump(mode="json")
        _write_result(output, record)
        raise
    artifact_store = ContentAddressedArtifactStore(Path("artifacts/compatibility/store"))
    artifact = _publish_record(artifact_store, probe, record)
    record["artifact"] = artifact.model_dump(mode="json")
    _write_result(output, record)


def _write_result(output: str, record: dict[str, Any]) -> None:
    encoded = json.dumps(record, sort_keys=True)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def _publish_record(
    artifact_store: ContentAddressedArtifactStore,
    probe: str,
    record: dict[str, Any],
) -> ArtifactRef:
    staged = artifact_store.stage_bytes(
        f"{probe}.json", canonical_bytes(record) + b"\n", producer="pre-activation-modal-probe"
    )
    return artifact_store.commit(staged, loader=lambda path: bool(json.loads(path.read_text())))


def _local_repository_revision(allow_dirty: bool) -> str:
    revision = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    if dirty and not allow_dirty:
        raise RuntimeError("canonical compatibility probes require a clean repository revision")
    return revision
