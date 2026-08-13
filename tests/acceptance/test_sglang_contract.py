from aurorapp.sglang_contract import greedy_generation_request


def test_pinned_raw_sglang_greedy_request_does_not_send_unsupported_seed() -> None:
    request = greedy_generation_request("hello", 32)

    assert request["sampling_params"] == {"temperature": 0, "max_new_tokens": 32}
