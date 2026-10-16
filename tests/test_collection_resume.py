"""A preempted collection must keep verified rows and safely finish its next attempt."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest



@pytest.mark.parametrize("generation_fails", [False, True])
@pytest.mark.parametrize("validation_count", [5, 10])
@pytest.mark.parametrize("finished_answer", [False, True])
def test_collection_resumes_without_repeating_verified_teachers(
    tmp_path,
    monkeypatch,
    finished_answer,
    validation_count,
    generation_fails,
    reject_next=False,
    require_full=False,
    no_saved=False,
):
    import subprocess

    import pipeline
    import reward
    import teacher

    seed = tmp_path / "seed"
    seed.mkdir()
    output = tmp_path / "run"
    (output / "data").mkdir(parents=True)

    def task(identity):
        return {"task_id": identity, "prompt": [], "length": "short"}

    def row(identity):
        return {
            **task(identity),
            "completion": [{"role": "assistant", "content": identity}],
            "reward": 0.7,
        }

    plan = {
        "run_name": pipeline.RUN_NAME,
        "run_id": pipeline.RUN_ID,
        "teacher_validation": True,
        "teacher_attempts": 1,
        "require_full_collection": require_full,
        "train_tasks": [task(f"t{i}") for i in range(20)],
        "validation_tasks": [task(f"v{i}") for i in range(validation_count)],
    }
    (seed / "plan.json").write_text(json.dumps(plan))
    (output / "plan.json").write_text(json.dumps(plan))
    (output / "status.json").write_text(
        json.dumps(
            {
                "activity": [{"at": "old", "message": "Previous progress"}],
                "acceptedTrain": 16,
                "acceptedValidation": 0,
            }
        )
    )
    (output / "data/train.jsonl").write_text(
        "".join(json.dumps(row(f"t{i}")) + "\n" for i in range(0 if no_saved else 16))
    )
    (output / "data/validation.jsonl").write_text("" if no_saved else json.dumps(row("v0")) + "\n")
    (tmp_path / "config.json").write_text(json.dumps({"model": "fixture"}))
    interrupted = output / "teacher/train/t16/0"
    interrupted.mkdir(parents=True)
    (interrupted / "prompt.txt").write_text("preserve this attempt")
    if finished_answer:
        (interrupted / "answer.txt").write_text("t16")
    real_path = Path

    def path(value):
        value = str(value)
        if value.startswith("/seed/"):
            return seed / value.removeprefix("/seed/")
        if value.startswith("/runs/"):
            return tmp_path / "published" / value.removeprefix("/runs/")
        return real_path(value)

    monkeypatch.setattr(pipeline, "Path", path)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "publish_collection", lambda *args: None)
    monkeypatch.setattr(pipeline, "directory", lambda: output)
    monkeypatch.setattr(pipeline, "results", SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(reward, "check_scorer", lambda: None)
    calls = []

    def generate(task, directory, **kwargs):
        calls.append(task["task_id"])
        if generation_fails:
            raise RuntimeError("fixture provider failure")
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "answer.txt").write_text(task["task_id"])
        return task["task_id"]

    monkeypatch.setattr(teacher, "generate", generate)
    monkeypatch.setattr(
        teacher,
        "verify",
        lambda task, text: (
            text,
            {
                "accepted": not (reject_next and task["task_id"] == "t16"),
                "reward": 0.7,
                "reason": "Requested 48 bars; counted 52"
                if reject_next and task["task_id"] == "t16"
                else None,
            },
        ),
    )
    if no_saved:
        with pytest.raises(RuntimeError, match="Need accepted teacher answers"):
            pipeline.collect()
        assert pipeline.status()["acceptedTrain"] == 0
        assert pipeline.status()["pendingTrain"] == 0
        assert len(json.loads((output / "data/rejected.json").read_text())) == 20 + validation_count
        return
    pipeline.collect()
    expected_calls = (
        ["t17", "t18", "t19"] if finished_answer else ["t16", "t17", "t18", "t19"]
    ) + [f"v{i}" for i in range(1, validation_count)]
    assert sorted(calls) == sorted(expected_calls)
    if generation_fails:
        assert len((output / "data/train.jsonl").read_text().splitlines()) == (
            17 if finished_answer else 16
        )
        assert pipeline.status()["state"] == "queued"
        assert pipeline.status()["pendingTrain"] == 0
        before = list(calls)
        pipeline.collect()
        assert calls == before
        return
    saved = [json.loads(line) for line in (output / "data/train.jsonl").read_text().splitlines()]
    assert [item["task_id"] for item in saved] == [
        f"t{i}" for i in range(20) if not (reject_next and i == 16)
    ]
    if reject_next:
        rejected = json.loads((output / "data/rejected.json").read_text())
        assert len(rejected) == 1 and rejected[0]["task_id"] == "t16"
        assert rejected[0]["reason"] == "Requested 48 bars; counted 52"
        assert (output / "teacher/train/t16/0/answer.txt").read_text() == "t16"
        assert pipeline.status()["rejectedTrain"] == 1
        assert pipeline.status()["state"] == "queued"
    assert len((output / "data/validation.jsonl").read_text().splitlines()) == validation_count
    assert pipeline.status()["acceptedValidation"] == validation_count
    assert pipeline.status()["activity"][0]["message"] == "Previous progress"
    if not finished_answer:
        assert list(interrupted.parent.glob("0-interrupted-*/prompt.txt"))
    before = list(calls)
    pipeline.collect()
    assert calls == before


def test_saved_rejection_continues_and_is_not_retried_on_resume(tmp_path, monkeypatch):
    test_collection_resumes_without_repeating_verified_teachers(
        tmp_path, monkeypatch, True, 5, False, reject_next=True
    )


def test_publication_does_not_rescore_completed_teachers(tmp_path, monkeypatch):
    import pipeline
    import recordings
    import teacher

    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model": "test", "revision": "fixed"}))
    public = tmp_path / "public"
    public.mkdir()
    existing = {
        "promptId": "t0",
        "split": "train",
        "version": "teacher",
        "reward": 0.7,
        "accepted": True,
    }
    (public / "evaluation.json").write_text(
        json.dumps({"verifierScores": [existing], "trainingRecordings": [], "recordings": []})
    )
    output = tmp_path / "run"
    (output / "data").mkdir(parents=True)
    (output / "data/train.jsonl").write_text(
        json.dumps({"task_id": "t0", "completion": [{"content": "abc"}]})
    )
    monkeypatch.setattr(
        pipeline, "Path", lambda value: public if str(value) == "/runs/public/runs" else Path(value)
    )
    # Production appends RUN_ID to the public root.
    monkeypatch.setattr(pipeline, "RUN_ID", "")
    monkeypatch.setattr(pipeline, "status", lambda: {"activity": []})

    def unexpected(*args, **kwargs):
        raise AssertionError("Already published teacher was scored again")

    monkeypatch.setattr(recordings, "public_recording", unexpected)
    monkeypatch.setattr(teacher, "verify", unexpected)
    pipeline.publish_collection(
        {"train_tasks": [{"task_id": "t0"}], "validation_tasks": [], "scorer_revision": "fixed"},
        output,
    )
    assert json.loads((public / "evaluation.json").read_text())["verifierScores"] == [existing]


def test_incomplete_collection_queues_accepted_teachers_even_with_old_full_flag(
    tmp_path, monkeypatch
):
    test_collection_resumes_without_repeating_verified_teachers(
        tmp_path, monkeypatch, True, 10, False, reject_next=True, require_full=True
    )


@pytest.mark.parametrize("has_answer", [False, True])
def test_failed_teacher_publication_survives_missing_answer_and_render_error(
    tmp_path, monkeypatch, has_answer
):
    import pipeline
    import recordings

    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model": "test", "revision": "fixed"}))
    public = tmp_path / "public"
    public.mkdir()
    output = tmp_path / "run"
    (output / "data").mkdir(parents=True)
    (output / "data/rejected.json").write_text(
        json.dumps(
            [
                {
                    "task_id": "bad",
                    "split": "train",
                    "attempt": 2,
                    "reward": 0,
                    "reason": "failed generation",
                }
            ]
        )
    )
    if has_answer:
        answer = output / "teacher/train/bad/2/answer.txt"
        answer.parent.mkdir(parents=True)
        answer.write_text("malformed")
    monkeypatch.setattr(
        pipeline, "Path", lambda value: public if str(value) == "/runs/public/runs" else Path(value)
    )
    monkeypatch.setattr(pipeline, "RUN_ID", "")
    monkeypatch.setattr(pipeline, "status", lambda: {"activity": []})

    def render(*args, **kwargs):
        assert has_answer
        raise ValueError("malformed music")

    monkeypatch.setattr(recordings, "public_recording", render)
    pipeline.publish_collection(
        {
            "train_tasks": [{"task_id": "bad", "prompt": []}],
            "validation_tasks": [],
            "scorer_revision": "fixed",
        },
        output,
    )
    row = json.loads((public / "evaluation.json").read_text())["verifierScores"][0]
    assert row["accepted"] is False and row["reward"] == 0


def test_zero_accepted_finishes_collection_but_cannot_train(tmp_path, monkeypatch):
    test_collection_resumes_without_repeating_verified_teachers(
        tmp_path, monkeypatch, False, 5, True, no_saved=True
    )
