"""Sequential account failover after explicit usage exhaustion; credentials stay private."""

import json
import os
import re
import threading
import time
from pathlib import Path


class AccountsUnavailable(RuntimeError):
    """Keep unfinished prompts pending until an account is available."""


class UsageLimitReached(RuntimeError):
    pass


def account_names():
    value = os.environ.get("MUSIC_CODEX_AUTH_VOLUMES") or os.environ.get(
        "MUSIC_CODEX_AUTH_VOLUME", "mimo-music-codex-auth"
    )
    names = [name.strip() for name in value.split(",")]
    if (
        not names
        or len(set(names)) != len(names)
        or any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name) for name in names)
    ):
        raise ValueError("Provide distinct comma-separated Codex auth volume names")
    return names


class AccountPool:
    def __init__(self, names=None, factory=None, commit=None, home=Path("/root/.codex")):
        self.names = names if names is not None else account_names()
        self.factory = factory or self._start
        self.commit = commit or (lambda name: None)
        self.home = home
        self.condition = threading.Condition()
        self.index = 0
        self.active = 0
        self.exhausted = set()
        self.server = None
        self.problem = None

    def _start(self, name):
        from codex_server import CodexServer

        target = Path("/auth") / name
        if not (target / "auth.json").is_file():
            raise AccountsUnavailable(f"Codex login missing from volume {name}")
        server = CodexServer(home=target)
        try:
            limits = server.rpc("account/rateLimits/read", {})
            buckets = limits.get("rateLimitsByLimitId") or {"codex": limits.get("rateLimits", {})}
            bucket = buckets.get("codex") or limits.get("rateLimits") or {}
            windows = [bucket.get("primary"), bucket.get("secondary")]
            exhausted = any(window and window.get("usedPercent", 0) >= 100 for window in windows)
            credits = bucket.get("credits") or {}
            if limits.get("ordinaryUsageAllowed") is False or (exhausted and not (credits.get("hasCredits") or credits.get("unlimited"))):
                raise UsageLimitReached("Teacher usage limit reached")
        except Exception:
            server.close()
            self.commit(name)
            raise
        return server

    def _close_server(self):
        if self.server is not None:
            self.server.close()
            self.server = None
            # Persist the refreshed credentials, never overwrite them from a seed.
            try:
                self.commit(self.names[self.index])
            except Exception as exc:
                self.problem = AccountsUnavailable("Could not persist refreshed Codex credentials")
                raise self.problem from exc

    def generate(self, prompt, directory):
        while True:
            with self.condition:
                if self.problem is not None:
                    raise self.problem
                while self.index < len(self.names) and self.names[self.index] in self.exhausted:
                    if self.problem is not None:
                        raise self.problem
                    if self.active:
                        self.condition.wait()
                        continue
                    self._close_server()
                    self.index += 1
                if self.index == len(self.names):
                    raise AccountsUnavailable("All configured Codex accounts are unavailable")
                name = self.names[self.index]
                if self.server is None:
                    try:
                        self.server = self.factory(name)
                    except Exception:
                        self.exhausted.add(name)
                        print("Teacher provider unavailable; trying the next configured account.", flush=True)
                        continue
                    print(f"Teacher Codex account: {name}", flush=True)
                server = self.server
                self.active += 1
            try:
                answer = server.generate(prompt, directory)
                return answer, name
            except UsageLimitReached:
                with self.condition:
                    if name not in self.exhausted:
                        print(
                            f"Codex usage limit: {name}; draining requests before switching.",
                            flush=True,
                        )
                    self.exhausted.add(name)
                # Quota does not consume one of the music-quality attempts.
                if directory.exists():
                    directory.rename(
                        directory.with_name(f"{directory.name}-quota-{time.time_ns()}")
                    )
                directory.mkdir(parents=True)
                (directory / "prompt.txt").write_text(prompt)
            finally:
                with self.condition:
                    self.active -= 1
                    self.condition.notify_all()

    def close(self):
        with self.condition:
            if self.active:
                raise RuntimeError("Cannot switch or close Codex while requests are active")
            self._close_server()

    def summary(self):
        return {"accounts": self.names, "exhausted": sorted(self.exhausted)}


def save_summary(pool, output):
    path = output / "codex-accounts.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(pool.summary(), indent=2) + "\n")
    temporary.replace(path)
