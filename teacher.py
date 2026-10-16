import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from codex_accounts import AccountPool, AccountsUnavailable, save_summary
from vendor.xiaomi_music_scorer import do, extract_abc

TEACHER = "gpt-6.1-sol"
INSTRUCTION = (
    "Generate the complete composition requested below. Return only plain ABC notation. "
    "Do not use tools, inspect files, or add commentary or Markdown fences. "
    "Use one tune, no blank lines, and explicit [V:id] sections for each voice. "
    "Write every requested bar in every voice, including rests; do not use repeats. "
    "Use valid ABC accidentals before notes.\n\n"
)


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def requested_bars(task):
    text = "\n".join(m["content"] for m in task["prompt"])
    match = re.search(r"(?:篇幅|about\s+)(\d+)\s*(?:小节|bars)", text)
    if not match:
        raise ValueError(f"Cannot find bar count: {task['task_id']}")
    return int(match.group(1))


def voice_bars(abc):
    """Count the same music in inline, standalone, and implicit ABC voices."""
    text = re.sub(r"(?m)%[^\n]*", "", abc)
    tune = re.split(r"(?m)^K:[^\n]*(?:\n|$)", text, maxsplit=1)
    header, body = (tune if len(tune) == 2 else ("", text))
    declared = re.findall(r"(?m)^V:([^\s]+)", header)
    voices = dict.fromkeys(declared, 0)
    body = re.sub(r"(?m)^\s*V:([^\s]+)[^\n]*$", r"[V:\1]", body)
    body = re.sub(r"(?m)^\s*[A-Za-z]:[^\n]*$", "", body)
    parts = re.split(r"\[V:([^\]\s]+)[^\]]*\]", body)

    def count(music):
        return len(re.findall(r"\|(?!\|)", music.replace("[|", "")))

    if parts[0].strip():
        initial = declared[0] if declared else "default"
        voices[initial] = count(parts[0])
    for voice, music in zip(parts[1::2], parts[2::2], strict=True):
        voices[voice] = voices.get(voice, 0) + count(music)
    return voices


def verify(task, answer):
    abc = extract_abc(answer)
    if not abc:
        return None, {"accepted": False, "reason": "no_abc"}
    abc = "\n".join(line for line in abc.splitlines() if line.strip()) + "\n"
    diagnostics = do(
        {
            "key": task["task_id"],
            "id": 0,
            "rep": 0,
            "abc": abc,
            "tag": task["tag"],
            "lang": task["lang"],
            "abc_len": len(abc),
            "nvoice": task["nvoice_want"],
            "latency": None,
        }
    )
    voices = voice_bars(abc)
    reward = (
        max(0.0, min(1.0, float(diagnostics["total"]) / 100.0))
        if "total" in diagnostics and not diagnostics.get("skip") and not diagnostics.get("reject")
        else 0.0
    )
    reasons = []
    if reward <= 0:
        reasons.append("No positive verifier reward")
    if diagnostics.get("err") != 0:
        reasons.append("ABC syntax errors")
    if diagnostics.get("bar") != 0:
        reasons.append("Invalid bar durations")
    if len(voices) != task["nvoice_want"]:
        reasons.append(f"Requested {task['nvoice_want']} voices; found {len(voices)}")
    expected = requested_bars(task)
    if any(bars != expected for bars in voices.values()):
        reasons.append(
            f"Requested {expected} bars per voice; counted "
            + ", ".join(f"{v}: {n}" for v, n in voices.items())
        )
    if re.search(r"\|:|:\|", abc):
        reasons.append("Repeats are not allowed")
    return abc, {
        "accepted": not reasons,
        "reason": "; ".join(reasons) if reasons else None,
        "reward": reward,
        "bars_by_voice": voices,
        "diagnostics": diagnostics,
    }


def generate(task, directory, accounts=None):
    if accounts is None:
        raise AccountsUnavailable("Teacher generation requires a managed Codex account pool")
    directory.mkdir(parents=True, exist_ok=False)
    prompt = INSTRUCTION + "\n".join(m["content"] for m in task["prompt"])
    (directory / "prompt.txt").write_text(prompt)
    started = time.monotonic()
    answer, account = accounts.generate(prompt, directory)
    (directory / "answer.txt").write_text(answer)
    save(
        directory / "generation.json",
        {
            "task": task,
            "model": TEACHER,
            "effort": "low",
            "account": account,
            "fresh_context": True,
            "prompt": prompt,
            "answer": answer,
            "answer_modified": False,
            "tool_calls": 0,
            "duration_ms": round((time.monotonic() - started) * 1000),
        },
    )
    return answer


def collect_one(task, output, split, attempts, accounts=None):
    """Workers own only their attempt directories; the caller saves shared state."""
    for attempt in range(attempts):
        directory = output / "teacher" / split / task["task_id"] / str(attempt)
        score_path = directory / "score.json"
        try:
            if score_path.exists():
                score = json.loads(score_path.read_text())
                if not score["accepted"]:
                    continue
            if (directory / "answer.txt").exists():
                answer = (directory / "answer.txt").read_text()
            else:
                if directory.exists():
                    directory.rename(directory.with_name(f"{attempt}-interrupted-{time.time_ns()}"))
                answer = generate(task, directory, accounts=accounts)
            abc, score = verify(task, answer)
        except AccountsUnavailable:
            raise
        except Exception as exc:
            # Keep provider diagnostics private; one bad request cannot abort collection.
            directory.mkdir(parents=True, exist_ok=True)
            score = {
                "accepted": False,
                "reward": 0,
                "reason": f"Generation or verification failed ({type(exc).__name__})",
            }
        save(score_path, score)
        if score["accepted"]:
            return {
                "prompt": task["prompt"],
                "completion": [{"role": "assistant", "content": abc}],
                "task_id": task["task_id"],
                "length": task["length"],
                "reward": score["reward"],
            }, None
    return None, {"task_id": task["task_id"], "split": split, "attempt": attempts - 1, **score}


def collect_pending(tasks, output, split, attempts=1, concurrency=10, accounts=None):
    if attempts < 1 or not 1 <= concurrency <= 10:
        raise ValueError("Collection requires positive attempts and 1-10 concurrent requests")
    owned = accounts is None
    accounts = accounts or AccountPool()
    deferred = None
    iterator = iter(tasks)
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = set()

            def submit():
                task = next(iterator, None)
                if task is not None:
                    pending.add(pool.submit(collect_one, task, output, split, attempts, accounts))

            for _ in range(concurrency):
                submit()
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    try:
                        yield future.result()
                    except AccountsUnavailable as exc:
                        deferred = exc
                # Drain already-running requests, but do not reject untouched prompts.
                if deferred is None:
                    for _ in finished:
                        submit()
        if deferred is not None:
            raise deferred
    finally:
        save_summary(accounts, output)
        if owned:
            accounts.close()
