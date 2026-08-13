import json
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field

from aurorapp.canonical import canonical_sha256
from aurorapp.models import JudgeCheck, JudgeResult, StrictModel


class JudgeRequest(StrictModel):
    check: JudgeCheck
    evidence: dict[str, Any]


class JudgeTransport(ABC):
    @abstractmethod
    def evaluate(self, request: JudgeRequest) -> dict[str, object]:
        raise NotImplementedError


class _JudgeAnswer(StrictModel):
    answer: str
    reason: str
    uncertainty: float = Field(ge=0, le=1)


class JudgeBroker:
    def __init__(self, transport: JudgeTransport) -> None:
        self.transport = transport
        self.cache: dict[str, JudgeResult] = {}

    def evaluate(self, request: JudgeRequest) -> JudgeResult:
        key = canonical_sha256(request)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        answers = tuple(
            _JudgeAnswer.model_validate(self.transport.evaluate(request)) for _ in range(3)
        )
        allowed = {"pass", "fail", "unclear"}
        if any(answer.answer not in allowed for answer in answers):
            raise ValueError("judge returned an answer outside the strict schema")
        majority = cast(
            Literal["pass", "fail", "unclear"],
            max(allowed, key=lambda value: sum(answer.answer == value for answer in answers)),
        )
        repeated = (answers[0].answer, answers[1].answer, answers[2].answer)
        result = JudgeResult(
            check_id=request.check.check_id,
            answer=majority,
            reason=answers[0].reason,
            uncertainty=max(answer.uncertainty for answer in answers),
            repeated_answers=repeated,
            source_evidence_hashes=[canonical_sha256(request.evidence)],
        )
        self.cache[key] = result
        return result


class CodexAppServerTransport(JudgeTransport):
    def __init__(self, codex_binary: Path, model: str) -> None:
        self.codex_binary = codex_binary
        self.model = model

    def evaluate(self, request: JudgeRequest) -> dict[str, object]:
        output_schema = _JudgeAnswer.model_json_schema()
        prompt = json.dumps(request.model_dump(mode="json"), sort_keys=True)
        with tempfile.TemporaryDirectory(prefix="aurorapp-judge-") as empty_directory:
            process = subprocess.Popen(
                [str(self.codex_binary), "app-server", "--stdio"],
                cwd=empty_directory,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                client = _AppServerClient(process, timeout_seconds=120)
                client.request(
                    "initialize",
                    {
                        "clientInfo": {"name": "aurorapp-judge", "version": "1"},
                        "capabilities": {},
                    },
                )
                client.notify("initialized", {})
                thread = client.request(
                    "thread/start",
                    {
                        "model": self.model,
                        "cwd": empty_directory,
                        "approvalPolicy": "never",
                        "sandbox": "read-only",
                        "ephemeral": True,
                        "baseInstructions": (
                            "Answer only the supplied binary evaluation check. Never call tools."
                        ),
                    },
                )
                turn = client.request(
                    "turn/start",
                    {
                        "threadId": thread["thread"]["id"],
                        "effort": "low",
                        "input": [{"type": "text", "text": prompt}],
                        "outputSchema": output_schema,
                    },
                )
                final_text = client.wait_for_turn(turn["turn"]["id"])
                decoded = json.loads(final_text)
                if not isinstance(decoded, dict):
                    raise RuntimeError("Codex judge response is not a JSON object")
                return decoded
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


class _AppServerClient:
    def __init__(self, process: subprocess.Popen[str], timeout_seconds: float) -> None:
        self.process = process
        self.timeout_seconds = timeout_seconds
        self.request_id = 0

    def _write(self, message: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Codex app-server stdin is unavailable")
        self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def _read(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("Codex app-server stdout is unavailable")
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if line:
                decoded = json.loads(line)
                if isinstance(decoded, dict):
                    return decoded
            if self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr is not None else ""
                raise RuntimeError(f"Codex app-server exited unexpectedly: {stderr[-4000:]}")
        raise TimeoutError("Codex app-server response timed out")

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"method": method, "params": params})

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.request_id += 1
        request_id = self.request_id
        self._write({"id": request_id, "method": method, "params": params})
        while True:
            message = self._read()
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(f"Codex app-server {method} failed: {message['error']}")
                result = message.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError(f"Codex app-server {method} returned no result object")
                return result
            if "id" in message and "method" in message:
                self._write(
                    {
                        "id": message["id"],
                        "error": {"code": -32601, "message": "no interactive capabilities"},
                    }
                )

    def wait_for_turn(self, turn_id: str) -> str:
        final_text: str | None = None
        while True:
            message = self._read()
            params = message.get("params", {})
            if not isinstance(params, dict):
                continue
            if message.get("method") == "item/completed" and params.get("turnId") == turn_id:
                item = params.get("item", {})
                if isinstance(item, dict) and item.get("type") == "agentMessage":
                    text = item.get("text")
                    if isinstance(text, str):
                        final_text = text
            if message.get("method") == "turn/completed":
                turn = params.get("turn", {})
                if isinstance(turn, dict) and turn.get("id") == turn_id:
                    if turn.get("status") != "completed":
                        raise RuntimeError(f"Codex judge turn failed: {turn.get('error')}")
                    if final_text is None:
                        raise RuntimeError("Codex judge completed without an agent message")
                    return final_text
