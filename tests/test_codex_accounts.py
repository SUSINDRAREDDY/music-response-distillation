import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import teacher
from codex_accounts import AccountPool, AccountsUnavailable, UsageLimitReached, account_names


def test_usage_limit_switch_drains_ten_requests_without_replaying_success(tmp_path):
    barrier = threading.Barrier(10, timeout=5)
    lock = threading.Lock()
    active = peak = 0
    opened = []
    closed = []
    calls = []
    commits = []

    class Server:
        def __init__(self, name):
            self.name = name

        def generate(self, prompt, directory):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                calls.append((self.name, prompt))
            try:
                if self.name == "first":
                    barrier.wait()
                    if prompt != "9":
                        raise UsageLimitReached()
                return prompt
            finally:
                with lock:
                    active -= 1

        def close(self):
            assert active == 0
            closed.append(self.name)

    def factory(name):
        assert active == 0
        opened.append(name)
        return Server(name)

    accounts = AccountPool(["first", "second"], factory=factory, commit=commits.append)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(accounts.generate, str(i), tmp_path / str(i)) for i in range(10)]
        results = [future.result(timeout=10) for future in futures]
    accounts.close()
    assert peak == 10
    assert opened == closed == commits == ["first", "second"]
    assert results[9] == ("9", "first")
    assert all(results[i] == (str(i), "second") for i in range(9))
    assert ("second", "9") not in calls
    assert accounts.exhausted == {"first"}


def test_nonquota_errors_never_switch_accounts(tmp_path):
    opened = []

    class Server:
        def generate(self, prompt, directory):
            raise RuntimeError("HTTP 429 temporary throughput limit")

        def close(self):
            pass

    accounts = AccountPool(
        ["first", "second"], factory=lambda name: opened.append(name) or Server()
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="throughput"):
            accounts.generate("prompt", tmp_path)
    accounts.close()
    assert opened == ["first"] and not accounts.exhausted


def test_exhaustion_keeps_prompts_pending_and_new_account_resumes(tmp_path, monkeypatch):
    calls = []

    class Server:
        def __init__(self, name):
            self.name = name

        def generate(self, prompt, directory):
            calls.append(self.name)
            if self.name != "new":
                (directory / "events.jsonl").write_text('{"type":"turn.failed"}\n')
                raise UsageLimitReached()
            return "ABC"

        def close(self):
            pass

    tasks = [{"task_id": str(i), "prompt": [], "length": "short"} for i in range(25)]
    monkeypatch.setattr(teacher, "verify", lambda *args: ("ABC", {"accepted": True, "reward": 0.5}))
    accounts = AccountPool(["old", "also-old"], factory=Server)
    with pytest.raises(AccountsUnavailable, match="All configured"):
        list(teacher.collect_pending(tasks, tmp_path, "train", accounts=accounts))
    accounts.close()
    assert len(calls) <= 20  # Only initially in-flight prompts can contact exhausted accounts.
    assert not list(tmp_path.glob("teacher/train/*/*/score.json"))
    assert list(tmp_path.glob("teacher/train/*/*-quota-*/events.jsonl"))
    accounts = AccountPool(["new"], factory=Server)
    outcomes = list(teacher.collect_pending(tasks, tmp_path, "train", accounts=accounts))
    accounts.close()
    assert len(outcomes) == 25 and all(row and failure is None for row, failure in outcomes)
    assert all((tmp_path / "teacher/train" / str(i) / "0/score.json").exists() for i in range(25))
    assert json.loads((tmp_path / "codex-accounts.json").read_text())["accounts"] == ["new"]


@pytest.mark.parametrize("value", ["a,a", "../auth", "a,", "a/b", ""])
def test_invalid_volume_selection_rejected(monkeypatch, value):
    monkeypatch.setenv("MUSIC_CODEX_AUTH_VOLUMES", value)
    monkeypatch.setenv("MUSIC_CODEX_AUTH_VOLUME", value)
    with pytest.raises(ValueError):
        account_names()


def test_startup_failure_tries_next_account(tmp_path):
    opened = []
    class Server:
        def generate(self, prompt, directory): return "ABC"
        def close(self): pass
    def factory(name):
        opened.append(name)
        if name == "broken": raise AccountsUnavailable("Login missing")
        return Server()
    pool = AccountPool(["broken", "working"], factory=factory)
    assert pool.generate("prompt", tmp_path) == ("ABC", "working")
    pool.close()
    assert opened == ["broken", "working"]

@pytest.mark.parametrize('used,allowed,blocked', [(100,False,True),(21,True,False)])
def test_live_limits_preflight_uses_explicit_account_home(monkeypatch, used, allowed, blocked):
    import codex_server
    from pathlib import Path
    homes=[];closed=[]
    class Server:
        def __init__(self, home): homes.append(str(home))
        def rpc(self, method, params):
            assert method == 'account/rateLimits/read'
            return {'ordinaryUsageAllowed':allowed,'rateLimitsByLimitId':{'codex':{'primary':{'usedPercent':used}}}}
        def close(self): closed.append(True)
    monkeypatch.setattr(codex_server,'CodexServer',Server)
    monkeypatch.setattr(Path,'is_file',lambda self:True)
    pool=AccountPool(['selected'])
    if blocked:
        with pytest.raises(UsageLimitReached):pool._start('selected')
        assert closed == [True]
    else: assert isinstance(pool._start('selected'),Server)
    assert homes == ['/auth/selected']
