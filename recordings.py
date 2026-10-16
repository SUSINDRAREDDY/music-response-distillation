"""Music scoring and publication helpers shared by training and inference."""

import json
import os
import re
import subprocess
import threading
from datetime import datetime, timezone

io_lock = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with io_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
        tmp.replace(path)


def track_gpu(stage):
    """Record each GPU invocation independently, including loading and waiting."""
    from functools import wraps

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            import inspect
            import time
            import uuid
            from pathlib import Path

            bound = inspect.signature(function).bind(*args, **kwargs)
            bound.apply_defaults()
            output = Path("/runs") / bound.arguments["run_name"]
            path = output / "gpu-runtime" / f"{stage}-{uuid.uuid4().hex}.json"
            module = inspect.getmodule(function)
            volume = module.results
            stop = threading.Event()
            started = time.monotonic()
            row = {"stage": stage, "gpu": "H100", "startedAt": now()}

            def persist(finished=False):
                try:
                    with io_lock:
                        value = {**row, "updatedAt": now(), "elapsedSeconds": time.monotonic() - started}
                        if finished:
                            value["endedAt"] = value["updatedAt"]
                        save(path, value)
                        volume.commit()
                except Exception:
                    # Cost telemetry must never interrupt paid training or inference.
                    import traceback
                    traceback.print_exc()

            def heartbeat():
                while not stop.wait(30):
                    persist()

            persist()
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                return function(*args, **kwargs)
            finally:
                stop.set()
                thread.join(timeout=35)
                persist(finished=True)
        return wrapped
    return decorate


def public_recording(
    task, text, version, run_id, directory, split, protocol, generation_abort=None
):
    """A bad composition must not discard the rest of an evaluation."""
    try:
        return _public_recording(task, text, version, run_id, directory, split, protocol, generation_abort)
    except (subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError, ZeroDivisionError) as exc:
        # Storage/engine errors still propagate; only sample rendering/scoring fails.
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "answer.txt").write_text(text)
        score = {"accepted": False, "reward": 0.0, "reason": f"Scoring/rendering failed: {type(exc).__name__}: {exc}"}
        save(directory / "score.json", score)
        return None, score


def _public_recording(task, text, version, run_id, directory, split, protocol, generation_abort):
    from teacher import verify
    from vendor.xiaomi_music_scorer import parse_midi

    if generation_abort:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "answer.txt").write_text(text)
        scored = {
            "accepted": False,
            "reward": 0.0,
            "reason": generation_abort,
            "generation_aborted": True,
        }
        save(directory / "score.json", scored)
        return None, scored
    abc, scored = verify(task, text)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "answer.txt").write_text(text)
    save(directory / "score.json", scored)
    if not abc:
        return None, scored
    (directory / "full.abc").write_text(abc)
    rendered = subprocess.run(
        [
            os.environ["ABC2MIDI_BIN"],
            str(directory / "full.abc"),
            "-o",
            str(directory / "full.mid"),
        ],
        capture_output=True,
        timeout=60,
    )
    midi = directory / "full.mid"
    parsed = parse_midi(midi.read_bytes()) if midi.exists() else None
    if (
        rendered.returncode
        or not parsed
        or not parsed["notes"]
        or scored.get("diagnostics", {}).get("err")
    ):
        return None, scored
    seconds = max(n[0] + n[1] for n in parsed["notes"]) / parsed["div"] * parsed["tempo"] / 1e6
    if seconds <= 0:
        return None, scored
    title = re.search(r"(?m)^T:(.*)$", abc)
    title = title.group(1).strip() if title else task["task_id"]
    voices = scored.get("bars_by_voice", {})
    bars = max(voices.values(), default=len(re.findall(r"\|(?!\|)", abc)))
    prompt = "\n".join(m["content"] for m in task["prompt"])
    header, *parts = re.split(r"(?=\[V:[^\]]+\])", abc)
    excerpt = header
    if parts:
        for part in parts:
            excerpt += "|".join(part.split("|")[:8]) + " |]\n"
    else:
        excerpt = "|".join(abc.split("|")[:8]) + " |]\n"
    save(directory / "details.json", {"prompt": prompt, "excerpt": excerpt})
    url = f"/music/runs/{run_id}/{split}/{task['task_id']}/{version}"
    score = {**protocol, "promptId": task["task_id"], "scope": "full-composition"}
    return {
        "id": task["task_id"],
        "title": title,
        "prompt": prompt,
        "meta": f"{task['bpm']} BPM · {task['meter']}",
        "length": task["length"],
        "split": split,
        "detailsUrl": url + "/details.json",
        "compositions": [
            {
                "version": version,
                "reward": scored.get("reward", 0),
                "scoring": score,
                "bars": bars,
                "voices": len(voices) or task["nvoice_want"],
                "seconds": seconds,
                "excerptSeconds": min(seconds, seconds * 8 / max(8, bars)),
                "fullUrl": url + "/full.abc",
                "detailsUrl": url + "/details.json",
            }
        ],
    }, scored
