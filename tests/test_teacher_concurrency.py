import json
import threading
from collections import Counter

import teacher


def test_ten_in_flight_mixed_failures_and_durable_attempts(tmp_path, monkeypatch):
    barrier = threading.Barrier(10, timeout=5)
    lock = threading.Lock()
    active = peak = 0
    calls = Counter()

    def generate(task, directory, **kwargs):
        nonlocal active, peak
        identity = task["task_id"]
        with lock:
            calls[identity] += 1
            active += 1
            peak = max(peak, active)
        try:
            if int(identity) < 10 and calls[identity] == 1:
                barrier.wait()
            directory.mkdir(parents=True)
            if identity == "3":
                raise TimeoutError("provider timed out")
            (directory / "answer.txt").write_text(identity)
            return identity
        finally:
            with lock:
                active -= 1

    def verify(task, answer):
        if answer == "4":
            raise ValueError("malformed music")
        return answer, {
            "accepted": answer != "5",
            "reward": 0.7,
            "reason": "bad bars" if answer == "5" else None,
        }

    monkeypatch.setattr(teacher, "generate", generate)
    monkeypatch.setattr(teacher, "verify", verify)
    tasks = [{"task_id": str(i), "prompt": [], "length": "short"} for i in range(24)]
    outcomes = list(teacher.collect_pending(tasks, tmp_path, "train"))
    assert peak == 10
    assert len(outcomes) == 24
    assert sum(row is not None for row, _ in outcomes) == 21
    assert {failure["task_id"] for _, failure in outcomes if failure} == {"3", "4", "5"}
    assert all(calls[str(i)] == 1 for i in range(24))
    assert json.loads((tmp_path / "teacher/train/3/0/score.json").read_text())["accepted"] is False
    before = calls.copy()
    # A crash before the collector journals results still must not regenerate attempts.
    resumed = list(teacher.collect_pending(tasks, tmp_path, "train"))
    assert calls == before
    assert sum(row is not None for row, _ in resumed) == 21
