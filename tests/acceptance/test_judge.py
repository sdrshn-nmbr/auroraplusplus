from aurorapp.judge import JudgeBroker, JudgeRequest, JudgeTransport
from aurorapp.models import JudgeCheck


class FakeTransport(JudgeTransport):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, request: JudgeRequest) -> dict[str, object]:
        self.calls += 1
        return {"answer": "fail", "reason": "missing evidence", "uncertainty": 0.1}


def test_judge_runs_check_three_times_and_caches_by_content() -> None:
    transport = FakeTransport()
    broker = JudgeBroker(transport)
    check = JudgeCheck(
        check_id="check",
        question="Is every timing window present?",
        required_evidence=["timing-windows"],
        version="1",
    )
    request = JudgeRequest(check=check, evidence={"timing-windows": []})

    first = broker.evaluate(request)
    second = broker.evaluate(request)

    assert first == second
    assert first.repeated_answers == ("fail", "fail", "fail")
    assert transport.calls == 3


def test_judge_is_advisory_and_has_no_activation_method() -> None:
    broker = JudgeBroker(FakeTransport())

    assert not hasattr(broker, "activate")
    assert not hasattr(broker, "promote")
