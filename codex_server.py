"""One Codex app-server multiplexes teacher turns through one managed auth session."""

import json
import os
import queue
import subprocess
import tempfile
import threading
import time

from codex_accounts import AccountsUnavailable, UsageLimitReached


def usage_exhausted(error):
    # Do not rotate for generic 429s, network errors, auth failures or access restrictions.
    message = error.get("message", "").casefold()
    return error.get("codexErrorInfo") == "usageLimitExceeded" and (
        "usage limit" in message or "out of credits" in message
    )


class CodexServer:
    def __init__(self, process=None, home=None):
        self.lock = threading.Lock()
        self.pending = {}
        self.threads = {}
        self.sequence = 0
        self.dead = False
        env = dict(os.environ)
        if home is not None:
            env["CODEX_HOME"] = str(home)
        for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"):
            env.pop(name, None)
        self.process = process or subprocess.Popen(
            [
                os.environ.get("CODEX_BIN", "codex"),
                "app-server",
                "--listen",
                "stdio://",
                "-c",
                'cli_auth_credentials_store="file"',
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.rpc("initialize", {"clientInfo": {"name": "mimo_music", "version": "1.0"}})
            self._send({"method": "initialized", "params": {}})
            account = self.rpc("account/read", {"refreshToken": False})
            if (account.get("account") or {}).get("type") != "chatgpt":
                raise AccountsUnavailable("The selected volume needs a ChatGPT device login")
        except Exception as exc:
            self.close()
            raise AccountsUnavailable(
                "Codex account initialization failed; check its login"
            ) from exc

    def _send(self, message):
        with self.lock:
            if self.dead:
                raise AccountsUnavailable("Codex app-server disconnected")
            try:
                self.process.stdin.write(json.dumps(message) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise AccountsUnavailable("Codex app-server disconnected") from exc

    def _read(self):
        try:
            for line in self.process.stdout:
                message = json.loads(line)
                if "id" in message and "method" in message:
                    # Teacher tasks never authorize tools, approvals, or interactive input.
                    self._send(
                        {
                            "id": message["id"],
                            "error": {"code": -32601, "message": "Teacher tools disabled"},
                        }
                    )
                    continue
                with self.lock:
                    if "id" in message:
                        channel = self.pending.get(message["id"])
                    else:
                        channel = self.threads.get(message.get("params", {}).get("threadId"))
                    if channel is not None:
                        channel.put(message)
        finally:
            with self.lock:
                self.dead = True
                for channel in list(self.pending.values()) + list(self.threads.values()):
                    channel.put(None)

    def rpc(self, method, params, timeout=30):
        with self.lock:
            self.sequence += 1
            identity = self.sequence
            channel = self.pending[identity] = queue.Queue()
        try:
            self._send({"id": identity, "method": method, "params": params})
            response = channel.get(timeout=timeout)
            if response is None:
                raise AccountsUnavailable("Codex app-server disconnected")
            if "error" in response:
                raise RuntimeError(f"Codex {method} failed")
            return response["result"]
        except queue.Empty as exc:
            raise AccountsUnavailable(f"Codex {method} timed out") from exc
        finally:
            with self.lock:
                self.pending.pop(identity, None)

    def generate(self, prompt, directory, timeout=900):
        directory.mkdir(parents=True, exist_ok=True)
        channel = queue.Queue()
        with tempfile.TemporaryDirectory(prefix="mimo-teacher-") as cwd:
            started = self.rpc(
                "thread/start",
                {
                    "model": "gpt-6.1-sol",
                    "cwd": cwd,
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    "ephemeral": True,
                    "config": {
                        "features.shell_tool": False,
                        "web_search": "disabled",
                        "project_doc_max_bytes": 0,
                        "model_reasoning_effort": "low",
                    },
                },
            )
            identity = started["thread"]["id"]
            with self.lock:
                self.threads[identity] = channel
            try:
                return self._turn(identity, channel, prompt, directory, timeout)
            finally:
                try:
                    self.rpc("thread/unsubscribe", {"threadId": identity})
                except (RuntimeError, AccountsUnavailable):
                    pass
                finally:
                    with self.lock:
                        self.threads.pop(identity, None)

    def _turn(self, identity, channel, prompt, directory, timeout):
        from teacher import save

        answer = None
        usage = None
        error = {}
        forbidden = False
        turn_id = None
        deadline = time.monotonic() + timeout
        with (directory / "events.jsonl").open("w") as log:

            def emit(value):
                log.write(json.dumps(value, ensure_ascii=False) + "\n")
                log.flush()

            emit({"type": "thread.started", "thread_id": identity})
            response = self.rpc(
                "turn/start",
                {
                    "threadId": identity,
                    "input": [{"type": "text", "text": prompt}],
                    "effort": "low",
                },
            )
            turn_id = response["turn"]["id"]
            while True:
                try:
                    message = channel.get(timeout=max(0, deadline - time.monotonic()))
                except queue.Empty as exc:
                    self.rpc("turn/interrupt", {"threadId": identity, "turnId": turn_id})
                    # Confirm cancellation before allowing this slot to issue another request.
                    cancel_deadline = time.monotonic() + 30
                    while True:
                        try:
                            cancelled = channel.get(
                                timeout=max(0, cancel_deadline - time.monotonic())
                            )
                        except queue.Empty as cancel_error:
                            raise AccountsUnavailable(
                                "Teacher cancellation was not confirmed"
                            ) from cancel_error
                        if cancelled is None:
                            raise AccountsUnavailable(
                                "Codex disconnected during cancellation"
                            ) from exc
                        if cancelled.get("method") == "turn/completed":
                            break
                    raise TimeoutError("Teacher turn timed out") from exc
                if message is None:
                    raise AccountsUnavailable("Codex app-server disconnected")
                params = message.get("params", {})
                method = message.get("method")
                if method == "thread/tokenUsage/updated":
                    total = params["tokenUsage"]["total"]
                    usage = {
                        field: total.get(key, 0)
                        for field, key in (
                            ("input_tokens", "inputTokens"),
                            ("cached_input_tokens", "cachedInputTokens"),
                            ("cache_write_input_tokens", "cacheWriteInputTokens"),
                            ("output_tokens", "outputTokens"),
                        )
                    }
                elif method in ("item/started", "item/completed"):
                    item = params["item"]
                    kind = item["type"]
                    if kind not in {"userMessage", "agentMessage", "reasoning", "plan"}:
                        forbidden = True
                    if method == "item/completed" and kind == "agentMessage":
                        answer = item["text"]
                        emit(
                            {
                                "type": "item.completed",
                                "item": {"type": "agent_message", "text": answer},
                            }
                        )
                elif method == "error":
                    error = params["error"]
                elif method == "turn/completed":
                    turn = params["turn"]
                    error = turn.get("error") or error
                    if usage is not None:
                        emit({"type": "turn.completed", "usage": usage})
                    if turn["status"] != "completed" or forbidden or not answer:
                        save(directory / "failure.json", {"error": error, "tool_calls": forbidden})
                        emit({"type": "turn.failed", "error": error})
                        if not forbidden and usage_exhausted(error):
                            raise UsageLimitReached("Codex account usage exhausted")
                        if error.get("codexErrorInfo") in ("unauthorized", "usageLimitExceeded"):
                            raise AccountsUnavailable(
                                "Codex account needs login or administrator action"
                            )
                        raise RuntimeError("Teacher turn failed; inspect private failure.json")
                    return answer

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.reader.join(timeout=5)
