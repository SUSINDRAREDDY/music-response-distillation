import json

import pytest

from data import ROOT, load_tasks, split_for
from reward import check_scorer, music_reward
from vendor.xiaomi_music_scorer import compute_score


def test_reward_matches_xiaomi():
    check_scorer()
    abc = (ROOT / "examples/smoke.abc").read_text()
    assert music_reward([abc]) == [compute_score("music", abc)]
    assert music_reward([[{"role": "assistant", "content": abc}]]) == music_reward([abc])


def test_invalid_answer_is_zero():
    assert music_reward(["No music notation here."]) == [0.0]


def test_missing_scorer_fails(monkeypatch):
    monkeypatch.setenv("ABC2MIDI_BIN", "/nonexistent/abc2midi")
    with pytest.raises(FileNotFoundError):
        check_scorer()


def test_dataset_splits():
    assert split_for("Compose  in E minor") == split_for("compose in e minor\n")
    rows = [load_tasks(ROOT / "data", name) for name in ("train", "validation", "test")]
    assert [len(split) for split in rows] == [806, 105, 89]
    ids = [{row["task_id"] for row in split} for split in rows]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert len(set.union(*ids)) == 1000
    prompts = [
        {" ".join(m["content"] for m in row["prompt"]).casefold() for row in split}
        for split in rows
    ]
    assert not prompts[0] & prompts[1]
    assert not prompts[0] & prompts[2]
    assert not prompts[1] & prompts[2]
    for split, expected in zip(("train", "validation", "test"), rows, strict=True):
        short = load_tasks(ROOT / "data", split, length="short")
        long = load_tasks(ROOT / "data", split, length="long")
        assert len(short) + len(long) == len(expected)
        assert {r["task_id"] for r in short}.isdisjoint(r["task_id"] for r in long)
        for length, subset in (("short", short), ("long", long)):
            saved = [
                json.loads(line)
                for line in (ROOT / "data" / f"{split}-{length}.jsonl").read_text().splitlines()
            ]
            assert saved == subset
