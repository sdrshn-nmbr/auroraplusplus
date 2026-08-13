import contextlib
import os
import signal
import socket
import subprocess
import sys

import pytest

from aurorapp.modal_probe import (
    _read_process_log,
    _specforge_ingest_after_prewarm,
    _start_logged_process,
    _stop_process_group,
    validate_dflash_capture_result,
    validate_specforge_ingest_result,
)
from aurorapp.sglang_contract import SGLANG_SERVER_RANDOM_SEED, greedy_generation_request


def test_pinned_raw_sglang_greedy_request_does_not_send_unsupported_seed() -> None:
    request = greedy_generation_request("hello", 32)

    assert request["sampling_params"] == {"temperature": 0, "max_new_tokens": 32}
    assert SGLANG_SERVER_RANDOM_SEED == 20260812


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
    }

    assert validate_specforge_ingest_result(result, [1, 2, 3, 4], [0, 0, 1, 1]) == (
        4,
        10240,
    )

    result["materialized"]["loss_mask"]["values"] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="loss mask"):
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
