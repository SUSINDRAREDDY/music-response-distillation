import json
from types import SimpleNamespace

import pytest

import pipeline

real_call_state = pipeline.call_state


@pytest.fixture(autouse=True)
def pending_call_status(monkeypatch):
    # Existing tests mock the call transport, not Modal's graph endpoint.
    monkeypatch.setattr(pipeline, "call_state", lambda call: getattr(call, "state", "PENDING"))


def test_preempted_coordinator_rejoins_existing_child(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        pipeline, "results", SimpleNamespace(reload=lambda: None, commit=lambda: None)
    )
    attempts = 0

    def get():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise KeyboardInterrupt("preempted coordinator")
        return {"optimizer_steps": 15}

    call = SimpleNamespace(object_id="fc-existing", get=get)

    def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        return call

    function = SimpleNamespace(spawn=spawn)
    monkeypatch.setattr(
        pipeline.modal,
        "FunctionCall",
        SimpleNamespace(from_id=lambda identity: call if identity == "fc-existing" else None),
    )
    with pytest.raises(KeyboardInterrupt):
        pipeline.durable_stage(tmp_path, "training", function, "run", "id", 4)
    assert pipeline.durable_stage(tmp_path, "training", function, "run", "id", 4) == {
        "optimizer_steps": 15
    }
    assert pipeline.durable_stage(tmp_path, "training", function, "run", "id", 4) == {
        "optimizer_steps": 15
    }
    assert len(calls) == 1
    assert attempts == 2
    with pytest.raises(ValueError, match="arguments differ"):
        pipeline.durable_stage(tmp_path, "training", function, "run", "id", 8)


def test_ambiguous_submission_never_launches_duplicate_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "results", SimpleNamespace(reload=lambda: None, commit=lambda: None)
    )
    (tmp_path / "orchestration.json").write_text(
        json.dumps({"evaluation": {"state": "launching", "request": {"args": [], "kwargs": {}}}})
    )
    with pytest.raises(RuntimeError, match="submission was interrupted"):
        pipeline.durable_stage(tmp_path, "evaluation", None)


def completed_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    config = {"model": "model", "revision": "fixed"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "plan.json").write_text(json.dumps({"run_id": "id", "epochs": 3}))
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                **config,
                "train_examples": 19,
                "settings": {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 4},
            }
        )
    )
    result = {"optimizer_steps": 15, "train": {"epoch": 3}}
    (tmp_path / "result.json").write_text(json.dumps(result))
    for step in (5, 10, 15):
        directory = tmp_path / f"checkpoint-{step}"
        directory.mkdir()
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            (directory / name).write_text("{}")
    return result


def test_completed_training_is_reused_and_mismatch_rejected(tmp_path, monkeypatch):
    result = completed_fixture(tmp_path, monkeypatch)
    assert pipeline.completed_training(tmp_path, "id", 4) == result
    with pytest.raises(ValueError):
        pipeline.completed_training(tmp_path, "other", 4)
    with pytest.raises(ValueError):
        pipeline.completed_training(tmp_path, "id", 8)
    (tmp_path / "checkpoint-10/adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        pipeline.completed_training(tmp_path, "id", 4)


def test_coordinator_skips_collection_and_training_after_completed_run(tmp_path, monkeypatch):
    completed_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "directory", lambda: tmp_path)
    monkeypatch.setattr(pipeline, "RUN_ID", "id")
    monkeypatch.setattr(pipeline, "GRAD_ACC", 4)
    monkeypatch.setattr(pipeline, "results", SimpleNamespace(reload=lambda: None))
    workers = []
    monkeypatch.setattr(
        pipeline.modal,
        "Function",
        SimpleNamespace(from_name=lambda *args: workers.append(args) or "inference"),
    )
    monkeypatch.setattr(pipeline, "publish", lambda *args, **kwargs: None)
    stages = []
    requests = []
    monkeypatch.setattr(
        pipeline,
        "durable_stage",
        lambda output, stage, *args, **kwargs: stages.append(stage) or requests.append(kwargs),
    )
    pipeline.coordinate.local()
    assert stages == ["evaluation"]
    assert workers == [("mimo-music-inference", "run")]
    assert requests[0]["concurrency"] == 64


def test_nonblocking_durable_stage_reuses_call_without_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "results", SimpleNamespace(reload=lambda: None, commit=lambda: None)
    )
    calls = []
    call = SimpleNamespace(object_id="fc-eval")
    function = SimpleNamespace(spawn=lambda: calls.append("spawn") or call)
    monkeypatch.setattr(
        pipeline.modal, "FunctionCall", SimpleNamespace(from_id=lambda identity: call)
    )
    assert pipeline.durable_stage(tmp_path, "evaluation", function, wait=False) is call
    assert pipeline.durable_stage(tmp_path, "evaluation", function, wait=False) is call
    assert calls == ["spawn"]


def test_waiting_account_collection_relaunches_only_the_finished_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "results", SimpleNamespace(reload=lambda: None, commit=lambda: None)
    )
    (tmp_path / "orchestration.json").write_text(
        json.dumps(
            {
                "collection": {
                    "state": "done",
                    "request": {"args": [], "kwargs": {}},
                    "result": {"state": "waiting_accounts"},
                    "call_id": "old",
                },
            }
        )
    )
    calls = []
    call = SimpleNamespace(object_id="new", get=lambda: {"state": "complete"})
    function = SimpleNamespace(spawn=lambda: calls.append("spawn") or call)
    assert pipeline.durable_stage(tmp_path, "collection", function) == {"state": "complete"}
    assert pipeline.durable_stage(tmp_path, "collection", function) == {"state": "complete"}
    assert calls == ["spawn"]


def test_waiting_accounts_never_allocates_training_or_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "directory", lambda: tmp_path)
    monkeypatch.setattr(pipeline, "results", SimpleNamespace(reload=lambda: None))
    calls = []

    def stage(output, name, *args, **kwargs):
        calls.append(name)
        return {"state": "waiting_accounts"}

    monkeypatch.setattr(pipeline, "durable_stage", stage)
    assert pipeline.coordinate.local() == {"state": "waiting_accounts"}
    assert calls == ["collection"]


def test_failed_child_restarts_once_and_keeps_call_history(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "results", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    def fail():
        raise RuntimeError("worker failed")
    old = SimpleNamespace(object_id="old", state="FAILURE", get=fail)
    new = SimpleNamespace(object_id="new", state="SUCCESS", get=lambda: "recovered")
    calls = iter([old, new])
    function = SimpleNamespace(spawn=lambda: next(calls))
    monkeypatch.setattr(pipeline.modal, "FunctionCall", SimpleNamespace(from_id=lambda _: old))
    with pytest.raises(RuntimeError, match="worker failed"):
        pipeline.durable_stage(tmp_path, "training", function)
    assert json.loads((tmp_path / "orchestration.json").read_text())["training"]["state"] == "failed"
    assert pipeline.durable_stage(tmp_path, "training", function) == "recovered"
    entry = json.loads((tmp_path / "orchestration.json").read_text())["training"]
    assert entry["previous_call_ids"] == ["old"]
    assert entry["state"] == "done"
    assert pipeline.durable_stage(tmp_path, "training", function) == "recovered"


def test_connection_error_does_not_mark_live_worker_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "results", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    def disconnected():
        raise ConnectionError("transport unavailable")
    call = SimpleNamespace(object_id="live", state="PENDING", get=disconnected)
    function = SimpleNamespace(spawn=lambda: call)
    with pytest.raises(ConnectionError):
        pipeline.durable_stage(tmp_path, "training", function)
    assert json.loads((tmp_path / "orchestration.json").read_text())["training"]["state"] == "running"


@pytest.mark.parametrize("failed_stage", ["training", "evaluation"])
def test_supervisor_handles_worker_failure(tmp_path, monkeypatch, failed_stage):
    monkeypatch.setattr(pipeline, "results", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    monkeypatch.setattr(pipeline, "publish", lambda *a, **k: None)
    cancelled = []
    calls = {name: SimpleNamespace(object_id=name, state="PENDING",
             cancel=lambda **kwargs: cancelled.append(kwargs)) for name in ("training", "evaluation")}
    rounds = {"training": 0, "evaluation": 0}
    def state(call):
        rounds[call.object_id] += 1
        if call.object_id == failed_stage:
            return "FAILURE"
        return "SUCCESS" if rounds[call.object_id] >= 2 else "PENDING"
    def finish(output, stage, call):
        if stage == failed_stage:
            raise RuntimeError(stage + " failed")
        return "saved"
    monkeypatch.setattr(pipeline, "call_state", state)
    monkeypatch.setattr(pipeline, "finish_stage", finish)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match=failed_stage + " failed"):
        pipeline.supervise_workers(tmp_path, **calls)
    if failed_stage == "training":
        assert cancelled == [{"terminate_containers": True}]
    else:
        assert not cancelled
        assert rounds["training"] == 2  # Training was allowed to finish.
        assert json.loads((tmp_path / "evaluation-status.json").read_text())["state"] == "failed"


def test_hard_training_failure_before_settings_never_launches_inference(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text(json.dumps({"stream_evaluation": True}))
    monkeypatch.setattr(pipeline, "directory", lambda: tmp_path)
    monkeypatch.setattr(pipeline, "results", SimpleNamespace(reload=lambda: None))
    monkeypatch.setattr(pipeline, "publish", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline.modal, "Function", SimpleNamespace(from_name=lambda *args: None))
    stages = []
    def stage(output, name, *args, **kwargs):
        stages.append(name)
        return SimpleNamespace(state="FAILURE") if name == "training" else None
    def failed(*args):
        raise RuntimeError("training container exited")
    monkeypatch.setattr(pipeline, "durable_stage", stage)
    monkeypatch.setattr(pipeline, "finish_stage", failed)
    with pytest.raises(RuntimeError, match="training container exited"):
        pipeline.coordinate_run()
    assert stages == ["collection", "preparation", "eligibility", "training"]


@pytest.mark.parametrize("error,expected", [
    (None, "SUCCESS"), (ValueError("remote failure"), "FAILURE"),
    (pipeline.modal.exception.TimeoutError(), "PENDING"),
    (pipeline.modal.exception.FunctionTimeoutError("worker timeout"), "FAILURE"),
    (pipeline.modal.exception.RemoteError("worker terminated"), "FAILURE"),
])
def test_result_poll_distinguishes_running_from_terminal(error, expected):
    def get(timeout):
        assert timeout == 0
        if error is not None:
            raise error
        return None
    assert real_call_state(SimpleNamespace(get=get)) == expected


@pytest.mark.parametrize("error", [ConnectionError("network"),
    pipeline.modal.exception.OutputExpiredError(), pipeline.modal.exception.AuthError("login")])
def test_unknown_call_outcome_never_allows_duplicate_work(error):
    def get(timeout):
        raise error
    with pytest.raises(type(error)):
        real_call_state(SimpleNamespace(get=get))
