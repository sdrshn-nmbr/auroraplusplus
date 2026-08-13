import argparse
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

INPUT_IDS = [1, 2, 3, 4]
LOSS_MASK = [0, 0, 1, 1]
AUX_LAYER_IDS = (1, 13, 25, 33, 39)


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
        target_model_version="poolside/Laguna-XS-2.1-INT4",
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
        store.release(handle, reason="compatibility-probe-consumed")
        released = True
        handle = None
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
            "released": released,
            "release_drain": release_drain,
        }
        print("AURORAPP_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    finally:
        if handle is not None:
            store.release(handle, reason="compatibility-probe-finally")
        elif ref is not None and not released:
            store.abort(ref.sample_id, reason="compatibility-probe-finally")
        if release_drain is None:
            store.drain_pending_removals()


if __name__ == "__main__":
    main()
