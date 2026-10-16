import json
from types import SimpleNamespace

import pytest
from data import filter_training_rows, training_data_path
from artifacts import combined_state


def test_overlength_teacher_is_excluded_without_truncating_other_answers(tmp_path):
    rows = [{"task_id": "fits", "prompt": [], "completion": [{"content": "keep"}]},
            {"task_id": "long", "prompt": [], "completion": [{"content": "too long"}]}]
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs:
        {"input_ids": [0] * (16384 if messages[0]["content"] == "keep" else 16385)})
    before = json.dumps(rows)
    kept, excluded = filter_training_rows(rows, tokenizer, 16384, {})
    assert [r["task_id"] for r in kept] == ["fits"]
    assert excluded[0]["task_id"] == "long"
    assert excluded[0]["tokens"] == 16385
    assert json.dumps(rows) == before
    assert training_data_path(tmp_path).name == "train.jsonl"
    (tmp_path / "train-sft.jsonl").write_text(json.dumps(kept[0]) + "\n")
    assert training_data_path(tmp_path).name == "train-sft.jsonl"


@pytest.mark.parametrize("training,evaluation,expected", [
    ("failed", "evaluating", "failed"), ("failed", "finished", "failed"),
    ("training", "failed", "failed"), ("training", "evaluating", "training"),
    ("evaluating", "evaluating", "evaluating"), ("evaluating", "finished", "finished"),
])
def test_failure_cannot_be_hidden_by_other_worker(training, evaluation, expected):
    assert combined_state(training, evaluation) == expected


@pytest.mark.parametrize("all_overlength", [False, True])
def test_cpu_preflight_preserves_originals_and_publishes_filtered_count(tmp_path, monkeypatch, all_overlength):
    import sys
    import pipeline
    rows = [{"task_id": "short", "prompt": [], "completion": [{"content": "short"}]},
            {"task_id": "long", "prompt": [], "completion": [{"content": "long"}]}]
    (tmp_path / "data").mkdir()
    source = tmp_path / "data/train.jsonl"
    original = "".join(json.dumps(row) + "\n" for row in rows)
    source.write_text(original)
    (tmp_path / "plan.json").write_text(json.dumps({"epochs": 3, "teacher_validation": False}))
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: {
        "input_ids": [0] * (16385 if all_overlength or messages[0]["content"] == "long" else 10)})
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)))
    monkeypatch.setattr(pipeline, "results", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    monkeypatch.setattr(pipeline, "cache", SimpleNamespace(commit=lambda: None))
    published = []
    monkeypatch.setattr(pipeline, "publish", lambda *args, **kwargs: published.append(kwargs))
    if all_overlength:
        with pytest.raises(ValueError, match="No eligible examples"):
            pipeline.prepare_training.local(str(tmp_path))
        assert published == []
    else:
        report = pipeline.prepare_training.local(str(tmp_path))
        assert report["train"]["kept"] == 1
        assert published[0]["trainingExamples"] == 1
        assert published[0]["totalSteps"] == 3
    assert source.read_text() == original
    filtered = (tmp_path / "data/train-sft.jsonl").read_text().splitlines()
    assert len(filtered) == (0 if all_overlength else 1)
    assert (tmp_path / "training-eligibility.json").exists()
