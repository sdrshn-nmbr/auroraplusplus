import contextlib
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

from aurorapp.canonical import canonical_sha256
from aurorapp.modal_probe import (
    _read_process_log,
    _run_until_terminal_record,
    _sampled_serving_arm,
    _specforge_ingest_after_prewarm,
    _start_logged_process,
    _stop_process_group,
    validate_dflash_capture_result,
    validate_specforge_ingest_result,
)
from aurorapp.sglang_contract import (
    SAMPLED_GENERATION_SEEDS,
    SGLANG_SERVER_RANDOM_SEED,
    MooncakeReplayEnvelope,
    PositionSamplingContract,
    SamplingDecisionDomain,
    greedy_generation_request,
    sampled_generation_request,
)


def test_pinned_raw_sglang_greedy_request_does_not_send_unsupported_seed() -> None:
    request = greedy_generation_request("hello", 32)

    assert request["sampling_params"] == {"temperature": 0, "max_new_tokens": 32}
    assert SGLANG_SERVER_RANDOM_SEED == 20260812


def test_pinned_raw_sglang_sampled_request_uses_request_sampling_seed() -> None:
    request = sampled_generation_request("hello", 32, sampling_seed=17)

    assert request["sampling_params"] == {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 32,
        "sampling_seed": 17,
    }
    assert "seed" not in request["sampling_params"]
    assert len(SAMPLED_GENERATION_SEEDS) == 3
    assert len(set(SAMPLED_GENERATION_SEEDS)) == 3


def test_target_and_dflash_use_the_same_position_sampling_decision() -> None:
    target = PositionSamplingContract(
        sampling_seed=17,
        absolute_position=41,
        domain=SamplingDecisionDomain.TARGET_TOKEN,
    )
    dflash = PositionSamplingContract(
        sampling_seed=17,
        absolute_position=41,
        domain=SamplingDecisionDomain.DFLASH_TARGET_TOKEN,
    )

    assert target.decision_key == dflash.decision_key
    assert target.model_copy(update={"absolute_position": 42}).decision_key != target.decision_key
    assert target.model_copy(update={"sampling_seed": 18}).decision_key != target.decision_key


def test_mooncake_replay_envelope_binds_all_derived_training_inputs() -> None:
    value = "a" * 64
    envelope = MooncakeReplayEnvelope(
        run_id=value,
        request_id="request-1",
        target_repository="poolside/Laguna-XS-2.1-INT4",
        target_revision="4b7e28abdc0a8b121def816b89d631750bc53c92",
        tokenizer_hash=value,
        sampling_contract_hash=value,
        input_ids_hash=value,
        loss_mask_hash=value,
        hidden_states_hash=value,
        hidden_state_shape=(1, 10, 10240),
        hidden_state_dtype="bfloat16",
        aux_layer_ids=(1, 13, 25, 33, 39),
    )

    assert envelope.hidden_state_shape == (1, 10, 10240)
    with pytest.raises(ValueError, match="official Laguna"):
        MooncakeReplayEnvelope.model_validate(
            {**envelope.model_dump(), "aux_layer_ids": [1, 13, 25]}
        )


@pytest.mark.parametrize("moe_runner_backend", [None, "triton"])
def test_sampled_probe_binds_deterministic_runtime(monkeypatch, moe_runner_backend) -> None:
    commands = []

    def start(command):
        commands.append(command)
        raise RuntimeError("stop after command construction")

    monkeypatch.setattr("aurorapp.modal_probe._start_logged_process", start)

    with pytest.raises(RuntimeError, match="command construction"):
        _sampled_serving_arm(
            port=30000,
            speculative_draft_path=None,
            moe_runner_backend=moe_runner_backend,
        )

    assert "--disable-radix-cache" in commands[0]
    assert "--enable-deterministic-inference" in commands[0]
    if moe_runner_backend is None:
        assert "--moe-runner-backend" not in commands[0]
    else:
        index = commands[0].index("--moe-runner-backend")
        assert commands[0][index + 1] == moe_runner_backend


def test_dflash_capture_contract_requires_all_official_laguna_layers() -> None:
    result = {
        "sample_id": "capture-1",
        "store_id": "compatibility",
        "gen": 1,
        "aux_layer_ids": [1, 13, 25, 33, 39],
        "features": {
            "target_aux_hidden_states": {
                "shape": [1, 7, 10240],
                "dtype": "bfloat16",
            }
        },
    }

    assert validate_dflash_capture_result(result) == (7, 10240)

    result["features"]["target_aux_hidden_states"]["shape"] = [1, 7, 8192]
    with pytest.raises(ValueError, match="feature shape"):
        validate_dflash_capture_result(result)


def test_specforge_ingest_requires_exact_materialized_training_tensors() -> None:
    value = "a" * 64
    result = {
        "sample_id": "aurorapp-ingest:task-1",
        "strategy": "dflash",
        "materialized": {
            "input_ids": {
                "shape": [1, 4],
                "dtype": "int64",
                "values": [1, 2, 3, 4],
            },
            "loss_mask": {
                "shape": [1, 4],
                "dtype": "int64",
                "values": [0, 0, 1, 1],
            },
            "hidden_states": {
                "shape": [1, 4, 10240],
                "dtype": "bfloat16",
                "finite": True,
                "absolute_sum": 12.5,
            },
        },
        "released": True,
        "release_drain": {"release_pending": 0},
        "replay_verified": True,
        "replay_envelope": {
            "run_id": value,
            "request_id": "task-1",
            "target_repository": "poolside/Laguna-XS-2.1-INT4",
            "target_revision": "4b7e28abdc0a8b121def816b89d631750bc53c92",
            "tokenizer_hash": value,
            "sampling_contract_hash": value,
            "input_ids_hash": canonical_sha256([1, 2, 3, 4]),
            "loss_mask_hash": canonical_sha256([0, 0, 1, 1]),
            "hidden_states_hash": value,
            "hidden_state_shape": [1, 4, 10240],
            "hidden_state_dtype": "bfloat16",
            "aux_layer_ids": [1, 13, 25, 33, 39],
        },
    }

    assert validate_specforge_ingest_result(result, [1, 2, 3, 4], [0, 0, 1, 1]) == (
        4,
        10240,
    )

    result["materialized"]["loss_mask"]["values"] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="loss mask"):
        validate_specforge_ingest_result(result, [1, 2, 3, 4], [0, 0, 1, 1])

    result["materialized"]["loss_mask"]["values"] = [0, 0, 1, 1]
    result["replay_verified"] = False
    with pytest.raises(ValueError, match="replay"):
        validate_specforge_ingest_result(result, [1, 2, 3, 4], [0, 0, 1, 1])


def test_specforge_reader_connects_only_after_capture_sink_is_warm(monkeypatch) -> None:
    events = []

    def capture(_port):
        events.append("capture-store-connected")
        return {"passed": True}

    def ingest(_port):
        assert events == ["capture-store-connected"]
        events.append("specforge-store-connected")
        return {"passed": True}

    monkeypatch.setattr("aurorapp.modal_probe._capture_generate", capture)
    monkeypatch.setattr("aurorapp.modal_probe._specforge_ingest", ingest)

    result = _specforge_ingest_after_prewarm(33000)

    assert result["passed"] is True
    assert result["prewarm"] == {"passed": True}
    assert events == ["capture-store-connected", "specforge-store-connected"]


def test_long_running_service_logs_cannot_fill_a_parent_pipe() -> None:
    process, log_path = _start_logged_process([sys.executable, "-c", "print('x' * 1_000_000)"])

    process.wait(timeout=5)
    output = _read_process_log(log_path)

    assert process.returncode == 0
    assert len(output) == 1_000_001


def test_terminal_record_ends_a_child_with_hanging_native_shutdown() -> None:
    started = time.monotonic()
    result = _run_until_terminal_record(
        [
            sys.executable,
            "-c",
            "import time; print('DONE={}', flush=True); time.sleep(60)",
        ],
        marker="DONE=",
        timeout=5,
    )

    assert time.monotonic() - started < 5
    assert result["terminal_records"] == ["{}"]
    assert result["terminated_after_terminal_record"] is True
    assert result["returncode"] == -signal.SIGTERM


def test_cleanup_kills_children_after_process_group_leader_exits(monkeypatch) -> None:
    monkeypatch.setattr("aurorapp.modal_probe._gpu_processes", lambda: [])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess; subprocess.Popen(['sleep', '60'])",
        ],
        start_new_session=True,
        text=True,
    )
    process_group = process.pid
    process.wait(timeout=5)
    try:
        cleanup = _stop_process_group(process, closed_port)
        assert cleanup["remaining_processes"] == []
        assert cleanup["port_closed"] is True
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group, signal.SIGKILL)
