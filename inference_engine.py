"""Continuous vLLM scheduling through its public LLMEngine API."""

import copy
import queue
import re
import threading
import time
from collections import defaultdict
from types import SimpleNamespace


def structural_overflow(text, expected_bars):
    """Conservative rejection, never a successful completion or repetition penalty.

    Count only completed lines of a single ABC tune. Leave ambiguous headers,
    multiple tunes to the full verifier. Support both ABC voice switch forms.
    Eight bars of slack avoid stopping on a partially formatted final phrase.
    """
    if not expected_bars or text.count("X:") != 1:
        return None
    completed = text.rsplit("\n", 1)[0] if "\n" in text else ""
    # Repeated MIDI directives contain no music and otherwise run to the cap.
    tail = [line.strip() for line in completed.splitlines() if line.strip()][-64:]
    if len(tail) == 64 and all(re.fullmatch(r"%%MIDI\s+.*", line) for line in tail):
        for period in range(1, 9):
            if all(line == tail[index % period] for index, line in enumerate(tail)):
                return "Repeated MIDI directives without note progress (64 lines); aborted generation loop"
    completed = re.sub(r"(?m)%[^\n]*", "", completed)
    # Empty voice switches or repeated lyric metadata can loop without bars.
    # Restrict this to non-musical lines; repeated notes are legitimate music.
    lines = [line.strip() for line in completed.splitlines() if line.strip()]
    tail = lines[-64:]
    if len(tail) == 64 and all(re.fullmatch(r"(?:\[V:[^\]]+\]|[Ww]:.*)", line) for line in tail):
        for period in range(1, 9):
            if all(line == tail[index % period] for index, line in enumerate(tail)):
                return "Repeated non-musical metadata without note progress (64 lines); aborted generation loop"
    # ABC permits standalone V: switches as well as inline [V:...] switches.
    completed = re.sub(r"(?m)^V:([^\s]+)[^\n]*$", r"[V:\1]", completed)
    parts = re.split(r"\[V:([^\]\s]+)[^\]]*\]", completed)
    counts = defaultdict(int)
    if len(parts) == 1:
        # With no voice switches, every explicit bar belongs to one voice.
        counts["default"] = len(re.findall(r"\|(?!\|)", completed))
    for voice, body in zip(parts[1::2], parts[2::2]):
        counts[voice] += len(re.findall(r"\|(?!\|)", body))
    for voice, bars in counts.items():
        if bars > expected_bars + 8:
            return f"Voice {voice} exceeded requested {expected_bars} bars ({bars} completed bars); aborted invalid generation"
    return None


class ContinuousEngine:
    def __init__(self, **settings):
        from vllm import EngineArgs, LLMEngine

        self.engine = LLMEngine.from_engine_args(
            EngineArgs(**settings), enable_multiprocessing=True
        )
        self.serial = 0
        self.capacity = settings.get("max_num_seqs", 64)

    def _iterate(self, prompts, sampling, lora_request=None, expected_bars=None, cancel=None):
        from vllm.sampling_params import RequestOutputKind

        params = copy.copy(sampling)
        params.output_kind = RequestOutputKind.CUMULATIVE
        active = {}
        self.serial += 1
        started = time.monotonic()
        first_tokens = {}
        first_token_seconds = 0.0
        checked = {}
        token_counts = {}
        total_tokens = 0
        self.telemetry = {"batchId": self.serial, "state": "generating", "outputTokens": 0}
        try:
            for index, prompt in enumerate(prompts):
                identity = f"{self.serial}-{index}"
                self.engine.add_request(identity, prompt, params, lora_request=lora_request)
                active[identity] = index
            while self.engine.has_unfinished_requests():
                if cancel is not None and cancel.is_set():
                    return
                for result in self.engine.step():
                    identity = result.request_id
                    if identity not in active:
                        continue
                    index = active[identity]
                    generated = result.outputs[0] if result.outputs else SimpleNamespace(
                        token_ids=[], text="", finish_reason="empty"
                    )
                    tokens = len(generated.token_ids)
                    total_tokens += tokens - token_counts.get(identity, 0)
                    token_counts[identity] = tokens
                    if tokens and identity not in first_tokens:
                        first_tokens[identity] = time.monotonic() - started
                        first_token_seconds += first_tokens[identity]
                    elapsed = time.monotonic() - started
                    self.telemetry = {"batchId": self.serial, "state": "generating", "outputTokens": total_tokens,
                        "elapsedSeconds": elapsed, "batchAverageTokensPerSecond": total_tokens / max(elapsed, .001),
                        "unfinishedRequests": len(active) - int(result.finished),
                        "completedRequests": len(prompts) - len(active) + int(result.finished),
                        "totalRequests": len(prompts),
                        "meanTimeToFirstTokenSeconds": first_token_seconds / len(first_tokens) if first_tokens else None}
                    reason = None
                    if (
                        expected_bars
                        and tokens - checked.get(identity, 0) >= 64
                        and not result.finished
                    ):
                        checked[identity] = tokens
                        reason = structural_overflow(generated.text, expected_bars[index])
                    if reason:
                        self.engine.abort_request([identity])
                    if result.finished or reason:
                        elapsed = time.monotonic() - started
                        del active[identity]
                        yield (
                            index,
                            result,
                            {
                                "tokens": tokens,
                                "request_seconds": elapsed,
                                "time_to_first_token_seconds": first_tokens.get(identity),
                                "finish_reason": "structure_abort"
                                if reason
                                else generated.finish_reason,
                                "abort_reason": reason,
                                "ended_with_eos": not reason and generated.finish_reason == "stop",
                            },
                        )
        finally:
            if active:
                self.engine.abort_request(list(active))
            self.telemetry = {**self.telemetry, "state": "scoring", "unfinishedRequests": 0}
        if active:
            raise RuntimeError("vLLM ended with incomplete requests")

    def generate(self, prompts, sampling_params, lora_request=None, use_tqdm=False):
        rows = list(self._iterate(prompts, sampling_params, lora_request))
        return [row[1] for row in sorted(rows, key=lambda row: row[0])]

    def completed(self, prompts, sampling, lora_request=None, expected_bars=None):
        """Pump GPU requests while the consumer scores and publishes on CPU.

        One owner accesses the engine. At most one final output per queued prompt
        is held in memory. Consumer errors cancel and join the producer.
        """
        messages = queue.Queue(maxsize=getattr(self, "capacity", 64))
        cancel = threading.Event()

        def send(message):
            while not cancel.is_set():
                try:
                    messages.put(message, timeout=0.1)
                    return
                except queue.Full:
                    pass

        def produce():
            try:
                for row in self._iterate(prompts, sampling, lora_request, expected_bars, cancel):
                    send(("result", row))
            except BaseException as exc:
                send(("error", exc))
            finally:
                send(("done", None))

        worker = threading.Thread(target=produce, name="music-vllm-scheduler", daemon=True)
        worker.start()
        try:
            while True:
                kind, payload = messages.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise payload
                yield payload
        finally:
            cancel.set()
            worker.join()
