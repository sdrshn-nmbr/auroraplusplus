import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import uuid
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from specforge.algorithms.common.dflash_family_data import build_collator
from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.inference.adapters.server_capture import (
    ServerCaptureFailure,
    ServerCaptureSchema,
    SGLangServerCaptureAdapter,
)
from specforge.inference.capture import CaptureConfig
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.runtime.contracts import PromptTask, TrainBatch
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.training.strategies.base import DFlashTrainStrategy
from transformers import AutoConfig

from aurorapp.model_port import (
    CapturedBatchOptimizerResult,
    CheckpointReferenceResult,
    CheckpointReloadResult,
)

INPUT_IDS = [1, 2, 3, 4]
LOSS_MASK = [0, 0, 1, 1]
AUX_LAYER_IDS = (1, 13, 25, 33, 39)
CHANGED_PARAMETER = "draft_model.layers.0.self_attn.g_proj.weight"
REQUIRED_GRADIENTS = (
    "draft_model.layers.0.self_attn.qkv_proj.weight",
    CHANGED_PARAMETER,
    "draft_model.fc.weight",
    "draft_model.aux_hidden_norms.0.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url")
    parser.add_argument("--target-repository")
    parser.add_argument("--target-revision")
    parser.add_argument("--draft-repository")
    parser.add_argument("--draft-revision")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--reload-checkpoint", type=Path)
    parser.add_argument("--reload-reference", type=Path)
    parser.add_argument("--write-reference", type=Path)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_draft_output(model: torch.nn.Module) -> torch.Tensor:
    torch.manual_seed(20260813)
    torch.cuda.manual_seed_all(20260813)
    noise = torch.randn(1, 4, 2048, device="cuda", dtype=torch.bfloat16)
    target_hidden = torch.randn(1, 2, 10240, device="cuda", dtype=torch.bfloat16)
    position_ids = torch.arange(6, device="cuda").unsqueeze(0)
    with torch.no_grad():
        return model(
            position_ids=position_ids,
            noise_embedding=noise,
            target_hidden=target_hidden,
        ).detach().cpu()


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(checkpoint: Path):
    config = AutoConfig.from_pretrained(checkpoint)
    config._attn_implementation = "eager"
    model, loading = AutoDraftModel.from_pretrained(
        checkpoint,
        config=config,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    return model.to("cuda").eval(), loading


def write_reference(checkpoint: Path, reference_path: Path) -> None:
    model, loading = load_checkpoint(checkpoint)
    if loading.get("missing_keys") or loading.get("unexpected_keys"):
        raise RuntimeError(f"reference checkpoint load is incomplete: {loading}")
    digest = state_digest(model)
    output = deterministic_draft_output(model)
    torch.save({"output": output, "state_digest": digest}, reference_path)
    result = CheckpointReferenceResult(
        state_digest=digest,
        reference_path=str(reference_path),
    )
    del model, output
    gc.collect()
    torch.cuda.empty_cache()
    print("AURORAPP_REFERENCE_RESULT=" + result.model_dump_json(), flush=True)


def reload_checkpoint(checkpoint: Path, reference_path: Path) -> None:
    model, loading = load_checkpoint(checkpoint)
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    expected = reference["output"]
    expected_state_digest = reference["state_digest"]
    observed_state_digest = state_digest(model)
    observed = deterministic_draft_output(model)
    difference = (expected.float() - observed.float()).abs()
    result = CheckpointReloadResult(
        missing=tuple(loading.get("missing_keys") or ()),
        unexpected=tuple(loading.get("unexpected_keys") or ()),
        state_digest=observed_state_digest,
        reference_state_digest=expected_state_digest,
        state_equal=observed_state_digest == expected_state_digest,
        output_equal=bool(torch.equal(expected, observed)),
        output_allclose=bool(torch.allclose(expected, observed, rtol=1e-5, atol=1e-6)),
        output_mismatch_count=int(torch.count_nonzero(expected != observed).item()),
        output_max_abs_difference=float(difference.max().item()),
        output_mean_abs_difference=float(difference.mean().item()),
    )
    del model, expected, observed, difference, reference
    gc.collect()
    torch.cuda.empty_cache()
    print("AURORAPP_RELOAD_RESULT=" + result.model_dump_json(), flush=True)


def feature_store() -> MooncakeFeatureStore:
    return MooncakeFeatureStore(
        store_id="aurorapp-specforge-captured-optimizer",
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


def capture_batch(base_url: str, store: MooncakeFeatureStore):
    task = PromptTask(
        task_id="captured-optimizer-task-1",
        run_id="aurorapp-captured-optimizer",
        source_id="compatibility-probe",
        payload={"input_ids": INPUT_IDS, "loss_mask": LOSS_MASK},
        max_length=len(INPUT_IDS),
        target_model_version="poolside/Laguna-XS-2.1-INT4",
        metadata={"num_tokens": len(INPUT_IDS), "tokenizer_version": "pinned"},
    )
    adapter = SGLangServerCaptureAdapter(
        base_url,
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
    refs = adapter.produce_refs([task], capture=capture)
    if len(refs) != 1:
        raise RuntimeError(f"SpecForge returned {len(refs)} refs for one task")
    ref = refs[0]
    if isinstance(ref, ServerCaptureFailure):
        raise RuntimeError(ref.reason)
    tensors, handle = store.get(ref)
    collated = build_collator()([tensors])
    batch = TrainBatch(
        sample_ids=[ref.sample_id],
        strategy="dflash",
        tensors=collated,
    )
    return ref, handle, batch


def build_training_model(args: argparse.Namespace):
    draft_config = AutoConfig.from_pretrained(
        args.draft_repository,
        revision=args.draft_revision,
    )
    draft_config._attn_implementation = "eager"
    draft_model, loading = AutoDraftModel.from_pretrained(
        args.draft_repository,
        revision=args.draft_revision,
        config=draft_config,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("unexpected_keys"):
        raise RuntimeError(f"official draft load is incomplete: {loading}")
    target_path = snapshot_download(
        repo_id=args.target_repository,
        revision=args.target_revision,
    )
    target_parts = TargetEmbeddingsAndHead.from_pretrained(
        target_path,
        device="cuda",
        dtype=torch.bfloat16,
    )
    method = draft_config.dflash_config
    training_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_parts.lm_head,
        target_embed_tokens=target_parts.embed_tokens,
        mask_token_id=int(method["mask_token_id"]),
        block_size=int(method["block_size"]),
        attention_backend="eager",
        num_anchors=1,
        objective_chunk_blocks=1,
        loss_type="dflash",
    ).to(device="cuda", dtype=torch.bfloat16)
    return training_model


def write_checkpoint(
    root: Path,
    model: OnlineDFlashModel,
    optimizer: torch.optim.Optimizer,
    sample_id: str,
) -> tuple[Path, dict[str, str]]:
    staging = root / ".staging" / str(uuid.uuid4())
    staging.mkdir(parents=True)
    model.draft_model.save_pretrained(staging, safe_serialization=True)
    optimizer_path = staging / "optimizer.pt"
    random_path = staging / "random-state.pt"
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(
        {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        },
        random_path,
    )
    paths = {
        "config": staging / "config.json",
        "weights": staging / "model.safetensors",
        "optimizer": optimizer_path,
        "random_state": random_path,
    }
    hashes = {name: file_hash(path) for name, path in paths.items()}
    manifest = {
        "files": hashes,
        "sample_id": sample_id,
        "training_cursor": 1,
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    hashes["manifest"] = file_hash(manifest_path)
    final = root / "objects" / hashes["manifest"][:2] / hashes["manifest"]
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise RuntimeError(f"checkpoint object already exists: {final}")
    staging.replace(final)
    return final, hashes


def parse_reload_result(output: str) -> CheckpointReloadResult:
    records = [
        line.removeprefix("AURORAPP_RELOAD_RESULT=")
        for line in output.splitlines()
        if line.startswith("AURORAPP_RELOAD_RESULT=")
    ]
    if len(records) != 1:
        raise RuntimeError(f"fresh reload returned {len(records)} terminal records: {output}")
    return CheckpointReloadResult.model_validate_json(records[0])


def train(args: argparse.Namespace) -> None:
    store = feature_store()
    ref = None
    handle = None
    released = False
    release_drain: dict[str, object] | None = None
    try:
        ref, handle, batch = capture_batch(args.base_url, store)
        training_model = build_training_model(args)
        strategy = DFlashTrainStrategy(training_model)
        training_model.train()
        trainable = [
            parameter for parameter in training_model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(trainable, lr=1e-5)
        parameters = dict(training_model.named_parameters())
        before = parameters[CHANGED_PARAMETER].detach().clone()
        optimizer.zero_grad(set_to_none=True)
        step = strategy.forward_loss(batch)
        step.loss.backward()
        gradients = tuple(
            name
            for name in REQUIRED_GRADIENTS
            if parameters[name].grad is not None
            and bool(torch.isfinite(parameters[name].grad).all().item())
        )
        optimizer.step()
        delta = float((before - parameters[CHANGED_PARAMETER].detach()).abs().sum().item())
        training_model.draft_model.eval()
        reference = deterministic_draft_output(training_model.draft_model)
        reference_state_digest = state_digest(training_model.draft_model)
        reference_path = Path("/tmp") / f"aurorapp-reload-{uuid.uuid4()}.pt"
        torch.save(
            {"output": reference, "state_digest": reference_state_digest},
            reference_path,
        )
        checkpoint, checkpoint_hashes = write_checkpoint(
            args.checkpoint_root,
            training_model,
            optimizer,
            ref.sample_id,
        )
        loss = float(step.loss.detach().item())
        accuracy = float(step.metrics["accuracy"].detach().item())
        accuracy_denom = int(step.metrics["accuracy_denom"].detach().item())
        optimizer_entries = len(optimizer.state)
        shapes = {name: tuple(tensor.shape) for name, tensor in batch.tensors.items()}
        del strategy, training_model, optimizer, parameters, before, reference, step, trainable
        gc.collect()
        torch.cuda.empty_cache()
        reload_process = subprocess.run(
            [
                sys.executable,
                __file__,
                "--reload-checkpoint",
                str(checkpoint),
                "--reload-reference",
                str(reference_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        reference_path.unlink(missing_ok=True)
        if reload_process.returncode != 0:
            raise RuntimeError(
                "fresh checkpoint reload failed: "
                + reload_process.stdout
                + reload_process.stderr
            )
        reload_result = parse_reload_result(reload_process.stdout)
        store.release(handle, reason="captured-optimizer-consumed")
        released = True
        handle = None
        release_drain = store.drain_pending_removals()
        result = CapturedBatchOptimizerResult(
            sample_id=ref.sample_id,
            input_ids_shape=shapes["input_ids"],
            loss_mask_shape=shapes["loss_mask"],
            hidden_states_shape=shapes["hidden_states"],
            loss=loss,
            accuracy=accuracy,
            accuracy_denom=accuracy_denom,
            gradient_parameters=gradients,
            optimizer_state_entries=optimizer_entries,
            changed_parameter=CHANGED_PARAMETER,
            parameter_delta=delta,
            checkpoint_hashes=checkpoint_hashes,
            checkpoint_path=str(checkpoint),
            training_cursor=1,
            reload_missing=reload_result.missing,
            reload_unexpected=reload_result.unexpected,
            reload_output_equal=reload_result.passed,
            released=released,
            release_pending=int(release_drain.get("release_pending", -1)),
        )
        if not result.passed or not math.isfinite(result.loss):
            raise RuntimeError(result.model_dump_json())
        print(
            "AURORAPP_RESULT="
            + result.model_dump_json(exclude={"passed"}),
            flush=True,
        )
    finally:
        if handle is not None:
            store.release(handle, reason="captured-optimizer-finally")
        elif ref is not None and not released:
            store.abort(ref.sample_id, reason="captured-optimizer-finally")
        if release_drain is None:
            store.drain_pending_removals()


def main() -> None:
    args = parse_args()
    if args.write_reference is not None:
        if args.reload_checkpoint is None:
            raise ValueError("reference mode requires --reload-checkpoint")
        write_reference(args.reload_checkpoint, args.write_reference)
        return
    if args.reload_checkpoint is not None:
        if args.reload_reference is None:
            raise ValueError("reload mode requires --reload-reference")
        reload_checkpoint(args.reload_checkpoint, args.reload_reference)
        return
    required = {
        "base_url": args.base_url,
        "target_repository": args.target_repository,
        "target_revision": args.target_revision,
        "draft_repository": args.draft_repository,
        "draft_revision": args.draft_revision,
        "checkpoint_root": args.checkpoint_root,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(f"training mode is missing arguments: {missing}")
    train(args)


if __name__ == "__main__":
    main()
