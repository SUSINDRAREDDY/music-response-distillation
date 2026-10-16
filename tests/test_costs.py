import json
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from costs import modal_usage, reset_run_costs, snapshot, teacher_usage


class CostsTest(unittest.TestCase):
    def test_cached_tokens_reasoning_duplicates_and_missing_reports(self):
        now = datetime.now(UTC)
        start = now - timedelta(hours=1)
        end = now + timedelta(hours=1)
        events = [
            {"type": "thread.started", "thread_id": "same-call"},
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 12055,
                    "cached_input_tokens": 9216,
                    "output_tokens": 918,
                    "reasoning_output_tokens": 59,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["0", "0-archived"]:
                path = root / "music-run/teacher/train/song" / name / "events.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text("\n".join(json.dumps(row) for row in events))
            missing = root / "music-run/teacher/train/failed/0/events.jsonl"
            missing.parent.mkdir(parents=True)
            missing.write_text('{"type":"turn.failed"}')
            old = root / "music-run/teacher/train/old/0/events.jsonl"
            old.parent.mkdir(parents=True)
            old.write_text(json.dumps(events[-1]))
            os.utime(old, (start.timestamp() - 1, start.timestamp() - 1))
            result = teacher_usage(root, start, end)
        self.assertEqual(result["total"], 12973)
        self.assertEqual(result["calls"], 1)
        self.assertEqual(result["missingCalls"], 1)
        self.assertAlmostEqual(result["estimatedCost"], 0.0157796)

    def test_long_context_pricing_applies_per_call(self):
        now = datetime.now(UTC)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "music-run/teacher/train/song/0/events.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 300000, "output_tokens": 1000},
                    }
                )
            )
            result = teacher_usage(root, now - timedelta(hours=1), now + timedelta(hours=1))
        self.assertAlmostEqual(result["estimatedCost"], 1.215)

    def test_modal_filters_other_apps_and_uses_workspace_rates(self):
        def row(name, env, resources):
            return SimpleNamespace(
                description=name, environment_name=env, cost_by_resource=resources
            )

        rows = [
            row("mimo-music-pipeline", "main", {"CPU": Decimal(".473"), "L40S": Decimal("3.9")}),
            row("unrelated", "main", {"CPU": Decimal("99")}),
            row("mimo-music-pipeline", "dev", {"CPU": Decimal("99")}),
            row("mimo-music-training-experiments", "main", {"CPU": Decimal("99")}),
        ]
        result = modal_usage(
            rows, {"cpu_hour_cost": Decimal(".0473"), "gpu_hour_cost_l40s": Decimal("1.95")}
        )
        self.assertAlmostEqual(result["reportedCost"], 4.373)
        self.assertEqual([r["hours"] for r in result["resources"]], [10, 2])


if __name__ == "__main__":
    unittest.main()


def test_inference_is_in_total_and_stage_breakdown_without_double_counting():
    rows = [
        SimpleNamespace(
            description=name, environment_name="main", cost_by_resource={"H100": Decimal(cost)}
        )
        for name, cost in [("mimo-music-inference", "2"), ("mimo-music-pipeline", "3")]
    ]
    result = modal_usage(rows, {"gpu_hour_cost_h100": Decimal("4")})
    assert result["reportedCost"] == 5
    assert sum(s["cost"] for s in result["stages"]) == 5
    assert next(s["cost"] for s in result["stages"] if s["name"] == "Inference") == 2


def test_run_cost_reset_preserves_history_and_resume_across_midnight(tmp_path, monkeypatch):
    import modal

    amount = Decimal("12")
    starts = []

    def report(*, start, end, resolution):
        starts.append(start)
        return [
            SimpleNamespace(
                description="mimo-music-pipeline",
                environment_name="main",
                cost_by_resource={"H100": amount},
            )
        ]

    workspace = SimpleNamespace(
        billing=SimpleNamespace(report=report, rates=lambda: {"gpu_hour_cost_h100": Decimal("4")})
    )
    monkeypatch.setattr(modal.Workspace, "from_context", lambda: workspace)
    output = tmp_path / "music-new"
    output.mkdir()
    reset_run_costs(output, "new")
    current = tmp_path / "public/live/current.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps({"runName": "music-new", "runId": "new"}))
    for run in ("music-old", "music-new"):
        event = tmp_path / run / "teacher/train/task/0/events.jsonl"
        event.parent.mkdir(parents=True)
        event.write_text(
            json.dumps(
                {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 10}}
            )
        )
        # Run totals include prior days of this run, not other runs from today.
        os.utime(event, (0, 0))
    value = snapshot(tmp_path)
    assert value["scope"] == "run" and value["model"]["calls"] == 1
    assert value["modal"]["reportedCost"] == 0
    baseline_path = output / "cost-baseline.json"
    baseline = json.loads(baseline_path.read_text())
    baseline["billingStart"] = (
        (datetime.now(UTC) - timedelta(days=1))
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .isoformat()
    )
    baseline_path.write_text(json.dumps(baseline))
    amount = Decimal("15")
    reset_run_costs(output, "new")
    assert json.loads(baseline_path.read_text()) == baseline
    value = snapshot(tmp_path)
    assert value["modal"]["reportedCost"] == 3
    assert value["modal"]["resources"][0]["hours"] == 0.75
    assert starts[-1] == datetime.fromisoformat(baseline["billingStart"])
    assert (tmp_path / "music-old/teacher/train/task/0/events.jsonl").exists()


def test_missing_baseline_never_shows_prior_daily_spend(tmp_path):
    current = tmp_path / "public/live/current.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps({"runName": "music-new", "runId": "new"}))
    value = snapshot(tmp_path)
    assert value["modal"] is None
    assert value["model"]["estimatedCost"] == 0


def test_gpu_workers_overlap_resume_and_stale_heartbeat(tmp_path):
    from costs import gpu_runtime
    now = datetime(2026, 10, 10, 12, tzinfo=UTC)
    directory = tmp_path / "gpu-runtime"
    directory.mkdir()
    def write(name, stage, seconds, age, ended=False):
        updated = (now - timedelta(seconds=age)).isoformat()
        row = {"stage": stage, "startedAt": (now - timedelta(hours=2)).isoformat(),
               "updatedAt": updated, "elapsedSeconds": seconds}
        if ended:
            row["endedAt"] = updated
        (directory / name).write_text(json.dumps(row))
    write("training-a.json", "training", 1800, 10)
    write("inference-a.json", "inference", 900, 300)
    write("inference-b.json", "inference", 900, 400, ended=True)
    rows = gpu_runtime(tmp_path, now, 4.0)
    assert [row["estimatedCost"] for row in rows] == [2.0, 2.0]
    assert [row["state"] for row in rows] == ["running", "unconfirmed"]
    # A crashed worker must not accumulate fictional hours after its last heartbeat.
    later = gpu_runtime(tmp_path, now + timedelta(days=1), 4.0)
    assert [row["estimatedCost"] for row in later] == [2.0, 2.0]
    assert gpu_runtime(tmp_path, now)[0]["estimatedCost"] is None


def test_gpu_runtime_closes_on_worker_failure(tmp_path, monkeypatch):
    import sys
    from recordings import track_gpu
    commits = []
    monkeypatch.setattr(sys.modules[__name__], "results",
                        SimpleNamespace(commit=lambda: commits.append(True)), raising=False)
    @track_gpu("training")
    def worker(run_name):
        raise ValueError("worker failed")
    try:
        worker(str(tmp_path))
    except ValueError:
        pass
    else:
        raise AssertionError("Worker error was swallowed")
    rows = list((tmp_path / "gpu-runtime").glob("*.json"))
    assert len(rows) == 1
    row = json.loads(rows[0].read_text())
    assert row["endedAt"] >= row["startedAt"]
    assert row["elapsedSeconds"] >= 0
    assert len(commits) == 2
