from typing import Any

SGLANG_SERVER_RANDOM_SEED = 20260812
SAMPLED_GENERATION_SEEDS = (17, 42, 20260812)
LAGUNA_TOKENIZER_FILE_HASHES = {
    "chat_template.jinja": "2d3c724b3c2e9eb71fe9ccc5423ff268a370a8bfa89e9238b6de14fe000825c8",
    "special_tokens_map.json": "70cd3459fde61761e9440751a590e89a108c09b1803cc7727f5ad1ed1ea6122b",
    "tokenizer.json": "807c53a95141e77c14e45f68c51db3f84d2ea6b555a6ea832bc99c88dae6a279",
    "tokenizer_config.json": "7458ce192bcd810dd43cad15c1996711168f2cc12d5400dac247da5033d6c283",
}


def greedy_generation_request(text: str, max_new_tokens: int) -> dict[str, Any]:
    if not text:
        raise ValueError("generation text cannot be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    return {
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }


def sampled_generation_request(
    text: str,
    max_new_tokens: int,
    *,
    sampling_seed: int,
) -> dict[str, Any]:
    if not text:
        raise ValueError("generation text cannot be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if sampling_seed < 0:
        raise ValueError("sampling_seed cannot be negative")
    return {
        "text": text,
        "sampling_params": {
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": max_new_tokens,
            "sampling_seed": sampling_seed,
        },
    }
