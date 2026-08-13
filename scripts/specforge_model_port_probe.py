import argparse
import gc
import json
import math

import torch
from specforge.modeling.auto import AutoDraftModel
from transformers import AutoConfig

from aurorapp.model_port import (
    LagunaDFlashConfigContract,
    PhysicalModelPortResult,
    validate_checkpoint_header,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = AutoConfig.from_pretrained(
        args.repository,
        revision=args.revision,
    )
    contract_config = LagunaDFlashConfigContract.model_validate_json(
        json.dumps(raw_config.to_dict())
    )
    raw_config._attn_implementation = "eager"
    model, loading = AutoDraftModel.from_pretrained(
        args.repository,
        revision=args.revision,
        config=raw_config,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    model = model.to("cuda")
    observed = {
        name: {"dtype": _dtype(value.dtype), "shape": list(value.shape)}
        for name, value in model.state_dict().items()
    }
    checkpoint = validate_checkpoint_header(contract_config, observed)

    torch.manual_seed(20260812)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    changed_name = "layers.0.self_attn.g_proj.weight"
    changed_parameter = dict(model.named_parameters())[changed_name]
    before = changed_parameter.detach().clone()
    noise_embedding = torch.randn(1, 4, 2048, device="cuda", dtype=torch.bfloat16)
    target_hidden = torch.randn(1, 2, 10240, device="cuda", dtype=torch.bfloat16)
    position_ids = torch.arange(6, device="cuda").unsqueeze(0)
    output = model(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
    )
    loss = output.float().square().mean()
    loss.backward()
    required_gradients = (
        "layers.0.self_attn.qkv_proj.weight",
        changed_name,
        "fc.weight",
        "aux_hidden_norms.0.weight",
    )
    parameters = dict(model.named_parameters())
    gradient_parameters = tuple(
        name
        for name in required_gradients
        if parameters[name].grad is not None
        and bool(torch.isfinite(parameters[name].grad).all().item())
    )
    optimizer.step()
    parameter_delta = float((before - changed_parameter.detach()).abs().sum().item())
    result = PhysicalModelPortResult(
        architecture=type(model).__name__,
        checkpoint=checkpoint,
        loading_missing=tuple(loading.get("missing_keys") or ()),
        loading_unexpected=tuple(loading.get("unexpected_keys") or ()),
        loading_mismatched=tuple(str(item) for item in loading.get("mismatched_keys") or ()),
        forward_shape=tuple(output.shape),
        loss_finite=math.isfinite(float(loss.detach().item())),
        gradient_parameters=gradient_parameters,
        optimizer_state_entries=len(optimizer.state),
        changed_parameter=changed_name,
        parameter_delta=parameter_delta,
    )
    if not result.passed:
        raise RuntimeError(result.model_dump_json())
    payload = result.model_dump_json(exclude={"passed"})
    del optimizer, output, loss, model
    gc.collect()
    torch.cuda.empty_cache()
    print("AURORAPP_RESULT=" + payload, flush=True)


def _dtype(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "BF16"
    return str(dtype).removeprefix("torch.").upper()


if __name__ == "__main__":
    main()
