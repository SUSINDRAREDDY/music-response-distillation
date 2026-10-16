import json
import queue
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from codex_accounts import UsageLimitReached
from codex_server import CodexServer, usage_exhausted


class FakeProcess:
    """Exercise actual JSON-RPC framing and notifications without using an account."""

    def __init__(self, error=None):
        self.error = error
        self.output = queue.Queue()
        self.stdout = iter(self.output.get, None)
        self.stdin = SimpleNamespace(write=self.write, flush=lambda: None)
        self.code = None
        self.messages = []

    def emit(self, message):
        self.output.put(json.dumps(message) + "\n")

    def write(self, text):
        message = json.loads(text)
        self.messages.append(message)
        method = message.get("method")
        if "id" not in message:
            return
        result = {}
        if method == "account/read":
            result = {"account": {"type": "chatgpt"}}
        elif method == "thread/start":
            assert message["params"]["sandbox"] == "read-only"
            assert message["params"]["ephemeral"] is True
            result = {"thread": {"id": f"thread-{message['id']}"}}
        elif method == "turn/start":
            result = {"turn": {"id": "turn-one"}}
        self.emit({"id": message["id"], "result": result})
        if method == "turn/start":
            identity = message["params"]["threadId"]
            self.emit(
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": identity,
                        "item": {"type": "reasoning", "text": "PRIVATE REASONING"},
                    },
                }
            )
            self.emit(
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": identity,
                        "tokenUsage": {
                            "total": {
                                "inputTokens": 100,
                                "cachedInputTokens": 20,
                                "cacheWriteInputTokens": 5,
                                "outputTokens": 10,
                            }
                        },
                    },
                }
            )
            if self.error is None:
                self.emit(
                    {
                        "method": "item/completed",
                        "params": {
                            "threadId": identity,
                            "item": {
                                "type": "agentMessage",
                                "text": message["params"]["input"][0]["text"],
                            },
                        },
                    }
                )
            self.emit(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": identity,
                        "turn": {
                            "id": "turn-one",
                            "status": "failed" if self.error else "completed",
                            "error": self.error,
                        },
                    },
                }
            )

    def poll(self):
        return self.code

    def terminate(self):
        self.code = 0
        self.output.put(None)

    def wait(self, timeout):
        return self.code


def test_app_server_turn_preserves_usage_and_discards_reasoning(tmp_path):
    process = FakeProcess()
    server = CodexServer(process=process)
    assert server.generate("compose music", tmp_path) == "compose music"
    server.close()
    text = (tmp_path / "events.jsonl").read_text()
    assert "PRIVATE REASONING" not in text
    usage = json.loads(text.splitlines()[-1])["usage"]
    assert usage == {
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "cache_write_input_tokens": 5,
        "output_tokens": 10,
    }
    assert any(m.get("method") == "thread/unsubscribe" for m in process.messages)


@pytest.mark.parametrize(
    "code,message,expected",
    [
        ("usageLimitExceeded", "You’ve hit your usage limit.", True),
        ("usageLimitExceeded", "Your workspace is out of credits.", True),
        ("rateLimitExceeded", "HTTP 429 slow down", False),
        ("unauthorized", "Sign in again", False),
        ("usageLimitExceeded", "This model is not included in your plan", False),
        ("usageLimitExceeded", "Spend cap set by your administrator", False),
        ("cyberPolicy", "Restricted", False),
        ("other", "You’ve hit your usage limit.", False),
    ],
)
def test_only_explicit_usage_exhaustion_triggers_failover(tmp_path, code, message, expected):
    error = {"codexErrorInfo": code, "message": message}
    assert usage_exhausted(error) is expected
    server = CodexServer(process=FakeProcess(error))
    try:
        with pytest.raises(UsageLimitReached if expected else RuntimeError):
            server.generate("compose", tmp_path)
    finally:
        server.close()
    assert json.loads((tmp_path / "failure.json").read_text())["error"] == error


def test_app_server_demultiplexes_concurrent_threads(tmp_path):
    server = CodexServer(process=FakeProcess())
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(server.generate, f"answer-{i}", tmp_path / str(i)) for i in range(20)
        ]
        assert [future.result(timeout=5) for future in futures] == [
            f"answer-{i}" for i in range(20)
        ]
    server.close()
    ids = [
        json.loads((tmp_path / str(i) / "events.jsonl").read_text().splitlines()[0])["thread_id"]
        for i in range(20)
    ]
    assert len(set(ids)) == 20
