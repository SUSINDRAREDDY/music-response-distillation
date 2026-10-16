import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_engine import ContinuousEngine, structural_overflow


def test_repeated_music_within_requested_bars_is_not_aborted():
    assert structural_overflow("X:1\n[V:1] " + ("C2 D2 | " * 32) + "\n", 32) is None


def test_guard_counts_complete_lines_across_voice_switches():
    text = (
        "X:1\n[V:1] "
        + ("C2 D2 | " * 25)
        + "\n[V:2] "
        + ("E2 F2 | " * 25)
        + "\n[V:1] "
        + ("G2 A2 | " * 25)
        + "\n"
    )
    assert "Voice 1" in structural_overflow(text, 32)
    assert structural_overflow(text.rstrip("\n"), 32) is None


def test_guard_ignores_comments_and_ambiguous_multiple_tunes():
    assert structural_overflow("X:1\n% [V:1] " + ("C | " * 100) + "\n", 32) is None
    assert structural_overflow("X:1\nX:2\n[V:1] " + ("C | " * 100) + "\n", 32) is None


class FakeEngine:
    def __init__(self, fail=False):
        self.requests = {}
        self.aborted = []
        self.iteration = 0
        self.fail = fail
        self.finished = threading.Event()

    def add_request(self, identity, prompt, params, **kwargs):
        self.requests[identity] = prompt

    def has_unfinished_requests(self):
        return bool(self.requests)

    def abort_request(self, identities):
        self.aborted.extend(identities)
        for identity in identities:
            self.requests.pop(identity, None)

    def step(self):
        if self.fail:
            raise ValueError("engine failed")
        # Finish the last submitted (short) prompt first, not submission order.
        identity = list(self.requests)[-1]
        text = self.requests.pop(identity)
        self.iteration += 1
        if not self.requests:
            self.finished.set()
        return [
            SimpleNamespace(
                request_id=identity,
                finished=True,
                outputs=[SimpleNamespace(text=text, token_ids=[1, 2], finish_reason="stop")],
            )
        ]


def make_engine(monkeypatch, fake=None):
    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(RequestOutputKind=SimpleNamespace(CUMULATIVE="cumulative")),
    )
    engine = ContinuousEngine.__new__(ContinuousEngine)
    engine.engine = fake or FakeEngine()
    engine.serial = 0
    return engine


def test_completion_order_and_cpu_scoring_do_not_block_decode(monkeypatch):
    engine = make_engine(monkeypatch)
    iterator = engine.completed(["long", "short"], SimpleNamespace())
    first = next(iterator)
    assert first[0] == 1
    # Producer finishes remaining work even while consumer has not requested it.
    assert engine.engine.finished.wait(2)
    assert next(iterator)[0] == 0
    with pytest.raises(StopIteration):
        next(iterator)


def test_probe_generate_restores_original_input_order(monkeypatch):
    engine = make_engine(monkeypatch)
    assert [r.outputs[0].text for r in engine.generate(["long", "short"], SimpleNamespace())] == [
        "long",
        "short",
    ]


def test_engine_error_cancels_outstanding_requests_and_reaches_consumer(monkeypatch):
    fake = FakeEngine(fail=True)
    engine = make_engine(monkeypatch, fake)
    with pytest.raises(ValueError, match="engine failed"):
        list(engine.completed(["a", "b"], SimpleNamespace()))
    assert len(fake.aborted) == 2
    assert not fake.requests


def test_abort_is_reported_as_failure_not_eos(monkeypatch):
    fake = FakeEngine()
    engine = make_engine(monkeypatch, fake)

    def step():
        identity = next(iter(fake.requests))
        return [
            SimpleNamespace(
                request_id=identity,
                finished=False,
                outputs=[
                    SimpleNamespace(
                        text="X:1\n[V:1] " + ("C | " * 50) + "\n",
                        token_ids=list(range(64)),
                        finish_reason=None,
                    )
                ],
            )
        ]

    fake.step = step
    rows = list(engine.completed(["music"], SimpleNamespace(), expected_bars=[32]))
    assert rows[0][2]["finish_reason"] == "structure_abort"
    assert rows[0][2]["ended_with_eos"] is False
    assert len(fake.aborted) == 1


def test_aborted_recording_is_saved_as_zero_reward_without_rendering(tmp_path):
    from recordings import public_recording

    row, score = public_recording(
        {},
        "partial music",
        "step-1",
        "run",
        tmp_path,
        "train",
        {},
        generation_abort="too many bars",
    )
    assert row is None
    assert score["reward"] == 0 and score["accepted"] is False
    assert (tmp_path / "answer.txt").read_text() == "partial music"
    assert not (tmp_path / "full.mid").exists()


def test_api_reads_selected_protocol_without_old_student_rewards(tmp_path):
    import json
    from artifacts import read_protocol_scores

    for namespace, reward in [("native-fp8", 0.9), ("continuous-test", 0)]:
        dest = tmp_path / namespace / "train" / "prompt" / "step-5"
        dest.mkdir(parents=True)
        (dest / "score.json").write_text(json.dumps({"reward": reward, "accepted": False}))
    teacher = {"promptId": "prompt", "split": "train", "version": "teacher", "reward": 0.8}
    scores = read_protocol_scores(
        tmp_path, {"artifactNamespace": "continuous-test", "verifierScores": [teacher]}
    )
    assert scores[0] == teacher
    assert len(scores) == 2 and scores[1]["reward"] == 0
    with pytest.raises(ValueError):
        read_protocol_scores(tmp_path, {"artifactNamespace": "../other"})


def test_metadata_loop_is_aborted_without_rejecting_repeated_notes():
    assert "metadata" in structural_overflow("X:1\n" + ("[V:1]\n[V:2]\n" * 32), 32)
    assert "metadata" in structural_overflow("X:1\n" + ("W:\nW: p\nW:\nW: mp\n" * 16), 32)
    assert structural_overflow("X:1\n[V:1]\n" + ("C D E F |\n" * 64), 64) is None


def test_telemetry_counts_output_tokens_once_and_tracks_completed_requests(monkeypatch):
    engine = make_engine(monkeypatch)
    list(engine.completed(["long", "short"], SimpleNamespace()))
    assert engine.telemetry["outputTokens"] == 4
    assert engine.telemetry["completedRequests"] == 2
    assert engine.telemetry["unfinishedRequests"] == 0
    assert engine.telemetry["totalRequests"] == 2
    assert engine.telemetry["batchAverageTokensPerSecond"] > 0
    assert engine.telemetry["state"] == "scoring"


@pytest.mark.parametrize('text', [
    'X:1\n' + '|: A,2 A,2 | A,2 A,2 | A,2 A,2 | A,2 A,2 :|\n' * 254,
    'X:1\nV:1 name="Piano"\n' + 'c2 c2 c2 c2 |\n' * 217,
    'X:1\n' + '%%MIDI program 0 0 46 46\n%%MIDI channel 0 1 2 3\n' * 288,
])
def test_actual_runaway_output_patterns_are_aborted(text):
    assert structural_overflow(text, 40)


def test_standalone_voice_switches_count_each_voice_separately():
    abc = 'X:1\nV:1\n' + 'C D |\n' * 40 + 'V:2\n' + 'E F |\n' * 40
    assert structural_overflow(abc, 40) is None


def test_bounded_queue_consumer_failure_does_not_deadlock(monkeypatch):
    engine = make_engine(monkeypatch)
    engine.capacity = 1
    iterator = engine.completed(['a', 'b', 'c', 'd'], SimpleNamespace())
    next(iterator)
    iterator.close()
    assert not any(t.name == 'music-vllm-scheduler' and t.is_alive() for t in threading.enumerate())


def test_render_timeout_becomes_failed_sample_and_next_output_continues(tmp_path, monkeypatch):
    import subprocess
    import recordings
    import teacher
    monkeypatch.setattr(teacher, 'verify', lambda *a: ('X:1\nK:C\nC|', {'accepted': True, 'reward': 1}))
    monkeypatch.setenv('ABC2MIDI_BIN', 'abc2midi')
    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired('abc2midi', 60)
    monkeypatch.setattr(recordings.subprocess, 'run', timeout)
    task = {'task_id': 'bad'}
    row, score = recordings.public_recording(task, 'saved answer', 'step-1', 'r', tmp_path/'bad', 'train', {})
    assert row is None and score['reward'] == 0 and not score['accepted']
    assert (tmp_path/'bad'/'answer.txt').read_text() == 'saved answer'
    monkeypatch.setattr(recordings, '_public_recording', lambda *a: ('next-row', {'reward': .5, 'accepted': True}))
    assert recordings.public_recording(task, 'next', 'step-1', 'r', tmp_path/'good', 'train', {})[0] == 'next-row'


def test_api_reuses_only_matching_protocol_index(tmp_path):
    import json
    from artifacts import read_protocol_scores

    directory = tmp_path / "selected"
    directory.mkdir()
    scores = [{"promptId": "p", "split": "train", "version": "step-1", "reward": 0}]
    (directory / "evaluation.json").write_text(json.dumps({
        "artifactNamespace": "selected", "reference": {"protocol": "a"}, "verifierScores": scores,
    }))
    assert read_protocol_scores(tmp_path, {
        "artifactNamespace": "selected", "reference": {"protocol": "a"},
    }) == scores
    assert read_protocol_scores(tmp_path, {
        "artifactNamespace": "selected", "reference": {"protocol": "b"},
    }) == []
