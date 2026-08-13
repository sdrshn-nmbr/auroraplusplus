import json
import os
import platform
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import modal

from aurorapp.artifacts import ContentAddressedArtifactStore
from aurorapp.canonical import canonical_bytes
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
    "git", "pciutils", "libcurl4", "libibverbs1", "librdmacm1"
)
base_image = runtime_base_image.add_local_python_source("aurorapp")

sglang_image = (
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
        "uv pip install --system mooncake-transfer-engine==0.3.12.post1",
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
    .add_local_python_source("aurorapp")
)

model_cache = modal.Volume.from_name("aurorapp-model-cache", create_if_missing=True)


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
            stdout, _ = process.communicate()
            raise RuntimeError(f"SGLang exited before health check:\n{stdout[-12000:]}")
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
    if process.poll() is None:
        os.killpg(process_group, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process_group, signal.SIGKILL)
            process.wait(timeout=30)
    remaining = subprocess.run(
        ["pgrep", "-g", str(process_group)],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "process_group": process_group,
        "remaining_processes": remaining,
        "gpu_processes": _gpu_processes(),
        "port_closed": _port_is_closed(port),
        "parent_returncode": process.returncode,
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
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
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
        stdout, _ = process.communicate()
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


def _capture_server_probe(repository_revision: str) -> dict[str, Any]:
    port = 32000
    os.environ.update(
        {
            "MOONCAKE_MASTER_SERVER_ADDR": "127.0.0.1:50051",
            "MOONCAKE_METADATA_SERVER": "http://127.0.0.1:8080/metadata",
            "MOONCAKE_LOCAL_HOSTNAME": "127.0.0.1",
            "MOONCAKE_PROTOCOL": "tcp",
        }
    )
    master = subprocess.Popen(
        ["mooncake_master", "--enable-http-metadata-server=true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
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
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    capture: dict[str, Any] = {"passed": False, "error": "capture did not run"}
    server_healthy = False
    error: dict[str, str] | None = None
    try:
        _wait_for_server(port, process, timeout=900)
        server_healthy = True
        capture = _capture_generate(port)
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    finally:
        server_cleanup = _stop_process_group(process, port)
        server_log, _ = process.communicate()
        master_cleanup = _stop_process_group(master, 50051)
        master_log, _ = master.communicate()
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
            if server_healthy and capture.get("passed") is True and cleanup_passed
            else "failed"
        ),
        "hardware": _hardware_identity(),
        "runtime": _runtime_identity(repository_revision),
        "server_healthy": server_healthy,
        "capture": capture,
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
    timeout=1800,
    volumes={"/root/.cache/huggingface": model_cache},
    single_use_containers=True,
    restrict_modal_access=True,
    name="laguna-dflash-capture-probe",
)
def laguna_dflash_capture_probe(repository_revision: str) -> dict[str, Any]:
    return _capture_server_probe(repository_revision)


@app.local_entrypoint()
def main(probe: str = "identity", output: str = "", allow_dirty: bool = False) -> None:
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
        else:
            raise ValueError("probe must be identity, target, dflash, or capture")
        remote_status = result.get("status") if isinstance(result, dict) else None
        record: dict[str, Any] = {
            "probe": probe,
            "status": remote_status or "passed",
            "result": result,
            "repository_revision": revision,
            "modal_image_id": (
                sglang_image.object_id if probe != "identity" else base_image.object_id
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
