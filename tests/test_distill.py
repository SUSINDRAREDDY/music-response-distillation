

from teacher import verify
from data import ROOT
from reward import music_reward


def test_teacher_filter_rejects_wrong_length_and_voice_count():
    task = {
        "task_id": "fixture",
        "prompt": [{"role": "user", "content": "about 8 bars"}],
        "tag": "fixture",
        "lang": "en",
        "nvoice_want": 1,
    }
    abc = (ROOT / "examples/smoke.abc").read_text().replace("K:Em\n", "K:Em\n[V:1]\n")
    cleaned, result = verify(task, abc.replace("[V:1]", "\n[V:1]\n"))
    assert result["accepted"]
    assert result["reward"] == music_reward([cleaned])[0]
    assert "\n\n" not in cleaned
    task["prompt"][0]["content"] = "about 9 bars"
    assert not verify(task, abc)[1]["accepted"]
    task["prompt"][0]["content"] = "about 8 bars"
    task["nvoice_want"] = 2
    assert not verify(task, abc)[1]["accepted"]


def test_teacher_filter_rejects_missing_music():
    assert verify({}, "No notation")[1]["accepted"] is False


def test_equivalent_voice_syntax_and_implicit_single_voice():
    abc = (ROOT / "examples/smoke.abc").read_text()
    task = {"task_id": "fixture", "prompt": [{"role": "user", "content": "about 8 bars"}],
            "tag": "fixture", "lang": "en", "nvoice_want": 1}
    scores = []
    for notation in (abc, abc.replace("K:Em\n", "K:Em\n[V:1]\n"),
                     abc.replace("K:Em\n", "K:Em\nV:1\n")):
        _, score = verify(task, notation)
        assert score["accepted"], score["reason"]
        assert list(score["bars_by_voice"].values()) == [8]
        scores.append(score["reward"])
    assert len(set(scores)) == 1
    task["nvoice_want"] = 2
    assert not verify(task, abc)[1]["accepted"]


def test_standalone_multi_voice_declarations_do_not_create_default_voice():
    from teacher import voice_bars
    text = "X:1\nV:RH name=Piano\nV:LH name=Piano\nK:C\nV:RH\nC4|D4|\nV:LH\nE4|F4|"
    assert voice_bars(text) == {"RH": 2, "LH": 2}


def test_first_declared_voice_is_implicit_after_key_header():
    from teacher import voice_bars
    text = "X:1\nV:RH name=Piano\nV:LH name=Piano\nK:C\nC4|D4|\nV:LH\nE4|F4|"
    assert voice_bars(text) == {"RH": 2, "LH": 2}
