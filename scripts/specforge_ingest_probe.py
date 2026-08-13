import argparse
import hashlib
import json
import os

import torch
from specforge.inference.adapters.server_capture import (
    ServerCaptureFailure,
    ServerCaptureSchema,
    SGLangServerCaptureAdapter,
)
from specforge.inference.capture import CaptureConfig
from specforge.runtime.contracts import PromptTask
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

from aurorapp.canonical import canonical_sha256
from aurorapp.sglang_contract import LAGUNA_TOKENIZER_FILE_HASHES, MooncakeReplayEnvelope

INPUT_IDS = [1, 2, 3, 4]
LOSS_MASK = [0, 0, 1, 1]
AUX_LAYER_IDS = (1, 13, 25, 33, 39)
TARGET_REPOSITORY = "poolside/Laguna-XS-2.1-INT4"
TARGET_REVISION = "4b7e28abdc0a8b121def816b89d631750bc53c92"


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()

    store = MooncakeFeatureStore(
        store_id="aurorapp-specforge-ingest",
        setup_kwargs={
            "local_hostname": os.environ["MOONCAKE_LOCAL_HOSTNAME"],
            "metadata_server": os.environ["MOONCAKE_METADATA_SERVER"],
            "global_segment_size": 1 << 30,
            "local_buffer_size": 1 << 30,
            "protocol": os.environ["MOONCAKE_PROTOCOL"],
            "rdma_devices": "",
            "master_server_addr": os.environ["MOONCAKE_MASTER_SERVER_ADDR"],
        },
    )
    task = PromptTask(
        task_id="task-1",
        run_id="aurorapp-ingest",
        source_id="compatibility-probe",
        payload={"input_ids": INPUT_IDS, "loss_mask": LOSS_MASK},
        max_length=len(INPUT_IDS),
        target_model_version=TARGET_REPOSITORY,
        metadata={"num_tokens": len(INPUT_IDS), "tokenizer_version": "pinned"},
    )
    adapter = SGLangServerCaptureAdapter(
        args.base_url,
        store,
        run_id=task.run_id,
        algorithm="dflash",
        schema=ServerCaptureSchema(
            aux_feature="hidden_states",
            last_hidden_feature=None,
            passthrough=(
                ("input_ids", "input_ids", ()),
                ("loss_mask", "loss_mask", ()),
            ),
        ),
        target_model_version=task.target_model_version,
    )
    capture = CaptureConfig.from_strategy(
        required_features={"input_ids", "loss_mask", "hidden_states"},
        aux_hidden_state_layer_ids=AUX_LAYER_IDS,
        target_repr=None,
        target_hidden_size=2048,
    )
    ref = None
    handle = None
    replay_handle = None
    released = False
    release_drain = None
    try:
        refs = adapter.produce_refs([task], capture=capture)
        if len(refs) != 1:
            raise RuntimeError(f"SpecForge returned {len(refs)} refs for one task")
        ref = refs[0]
        if isinstance(ref, ServerCaptureFailure):
            raise RuntimeError(ref.reason)
        tensors, handle = store.get(ref)
        replay_tensors, replay_handle = store.get(ref)
        tensor_hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
        replay_tensor_hashes = {
            name: tensor_sha256(value) for name, value in replay_tensors.items()
        }
        replay_verified = tensor_hashes == replay_tensor_hashes
        if not replay_verified:
            raise RuntimeError("Mooncake replay returned different tensor bytes")
        materialized = {
            "input_ids": {
                "shape": list(tensors["input_ids"].shape),
                "dtype": str(tensors["input_ids"].dtype).removeprefix("torch."),
                "values": tensors["input_ids"].reshape(-1).tolist(),
            },
            "loss_mask": {
                "shape": list(tensors["loss_mask"].shape),
                "dtype": str(tensors["loss_mask"].dtype).removeprefix("torch."),
                "values": tensors["loss_mask"].reshape(-1).tolist(),
            },
            "hidden_states": {
                "shape": list(tensors["hidden_states"].shape),
                "dtype": str(tensors["hidden_states"].dtype).removeprefix("torch."),
                "finite": bool(torch.isfinite(tensors["hidden_states"]).all().item()),
                "absolute_sum": float(tensors["hidden_states"].float().abs().sum().item()),
            },
        }
        envelope = MooncakeReplayEnvelope(
            run_id=canonical_sha256(
                {
                    "probe": "specforge-ingest",
                    "target_repository": TARGET_REPOSITORY,
                    "target_revision": TARGET_REVISION,
                    "task": task.payload,
                }
            ),
            request_id=task.task_id,
            target_repository=TARGET_REPOSITORY,
            target_revision=TARGET_REVISION,
            tokenizer_hash=canonical_sha256(LAGUNA_TOKENIZER_FILE_HASHES),
            sampling_contract_hash=canonical_sha256(
                {"version": "capture-only-v1", "sampling": "none"}
            ),
            input_ids_hash=canonical_sha256(INPUT_IDS),
            loss_mask_hash=canonical_sha256(LOSS_MASK),
            hidden_states_hash=tensor_hashes["hidden_states"],
            hidden_state_shape=tuple(
                int(value) for value in tensors["hidden_states"].shape
            ),
            hidden_state_dtype="bfloat16",
            aux_layer_ids=AUX_LAYER_IDS,
        )
        store.release(handle, reason="compatibility-probe-consumed")
        handle = None
        store.release(replay_handle, reason="compatibility-probe-replay-consumed")
        replay_handle = None
        released = True
        release_drain = store.drain_pending_removals()
        result = {
            "sample_id": ref.sample_id,
            "strategy": ref.strategy,
            "feature_store_uri": ref.feature_store_uri,
            "feature_specs": {
                name: {
                    "shape": list(spec.shape),
                    "dtype": spec.dtype,
                }
                for name, spec in ref.feature_specs.items()
            },
            "materialized": materialized,
            "replay_envelope": envelope.model_dump(mode="json"),
            "replay_verified": replay_verified,
            "released": released,
            "release_drain": release_drain,
        }
        print("AURORAPP_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    finally:
        if handle is not None:
            store.release(handle, reason="compatibility-probe-finally")
        if replay_handle is not None:
            store.release(replay_handle, reason="compatibility-probe-replay-finally")
        if ref is not None and not released:
            store.abort(ref.sample_id, reason="compatibility-probe-finally")
        if release_drain is None:
            store.drain_pending_removals()


if __name__ == "__main__":
    main()
