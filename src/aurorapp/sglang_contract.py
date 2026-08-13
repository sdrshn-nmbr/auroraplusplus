from typing import Any


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
