import contextlib
import os
import signal
import socket
import subprocess
import sys

import pytest

from aurorapp.modal_probe import _stop_process_group, validate_dflash_capture_result
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
