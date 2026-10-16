"""Detached verified teacher collection, TRL training, and held-out vLLM evaluation."""

import json
import math
import os
import time
from pathlib import Path

import modal
from codex_accounts import AccountsUnavailable, account_names
from recordings import track_gpu

ROOT = Path(__file__).resolve().parent
RUN_NAME = os.environ.get("MUSIC_RUN_NAME", "")
RUN_ID = os.environ.get("MUSIC_RUN_ID", "")
GRAD_ACC = json.loads((ROOT / "config.json").read_text())["sft"]["trainer"][
    "gradient_accumulation_steps"
]
BATCH_SIZE = json.loads((ROOT / "config.json").read_text())["sft"]["trainer"]["per_device_train_batch_size"]

def optimizer_steps_per_epoch(examples, batch_size, accumulation):
    return math.ceil(math.ceil(examples / batch_size) / accumulation)

if GRAD_ACC < 1:
    raise ValueError("Gradient accumulation must be positive")
app = modal.App("mimo-music-pipeline")
results = modal.Volume.from_name("mimo-music-runs")
cache = modal.Volume.from_name("mimo-music-model-cache")
auth_volumes = {name: modal.Volume.from_name(name) for name in account_names()}
run_environment = {
    "MUSIC_RUN_NAME": RUN_NAME,
    "MUSIC_RUN_ID": RUN_ID,
    "MUSIC_CODEX_AUTH_VOLUME": os.environ.get("MUSIC_CODEX_AUTH_VOLUME", "mimo-music-codex-auth"),
    "MUSIC_CODEX_AUTH_VOLUMES": ",".join(auth_volumes),
}
base = modal.Image.debian_slim(python_version="3.12")
training_image = base
collection_image = base
coordinator_image = base
tokenizer_image = base
if modal.is_local() and RUN_NAME:
    from sft import image as original_training_image

    training_image = original_training_image.add_local_file(
        ROOT / "pipeline.py", "/root/pipeline.py"
    )
    collection_image = (
        base.apt_install("nodejs", "npm", "build-essential")
        .run_commands("npm install -g @openai/codex@0.160.0")
        .pip_install("modal==1.6.1")
        .add_local_file(ROOT / "setup_scorer.py", "/root/setup_scorer.py", copy=True)
        .run_commands("python /root/setup_scorer.py")
        .env({"ABC2MIDI_BIN": "/root/tools/abc2midi", "PYTHONPATH": "/root", **run_environment})
        .add_local_file(ROOT / "runs" / RUN_NAME / "plan.json", "/seed/plan.json")
        .add_local_dir(ROOT / "vendor", "/root/vendor", ignore=["**/__pycache__/**"])
    )
    for name in (
        "pipeline.py",
        "teacher.py",
        "codex_accounts.py",
        "codex_server.py",
        "costs.py",
        "data.py",
        "reward.py",
        "recordings.py",
        "config.json",
    ):
        collection_image = collection_image.add_local_file(ROOT / name, "/root/" + name)
    coordinator_image = (
        base.pip_install("modal==1.6.1")
        .env(run_environment)
        .add_local_file(ROOT / "pipeline.py", "/root/pipeline.py")
        .add_local_file(ROOT / "codex_accounts.py", "/root/codex_accounts.py")
        .add_local_file(ROOT / "recordings.py", "/root/recordings.py")
        .add_local_file(ROOT / "config.json", "/root/config.json")
    )

    tokenizer_image = (
        base.pip_install("modal==1.6.1", "transformers==5.18.0", "jinja2==3.1.6")
        .env({**run_environment, "HF_HOME": "/cache/huggingface"})
    )
    for name in ("pipeline.py", "codex_accounts.py", "recordings.py", "config.json", "data.py"):
        tokenizer_image = tokenizer_image.add_local_file(ROOT / name, "/root/" + name)


def directory():
    return Path("/runs") / RUN_NAME


def status():
    return json.loads((directory() / "status.json").read_text())


def publish(message=None, run_name=None, **updates):
    from recordings import now, save

    output = Path("/runs") / run_name if run_name else directory()
    value = json.loads((output / "status.json").read_text())
    value.update(updates)
    if message:
        value["activity"].append({"at": now(), "message": message})
        print(message, flush=True)
    value["updatedAt"] = now()
    save(output / "status.json", value)
    save(Path("/runs/public/live/current.json"), value)
    results.commit()
    return value


def completed_training(output, run_id, gradient_accumulation):
    """Reuse a verified completed run before allocating another training GPU."""
    result_path = output / "result.json"
    if not result_path.exists():
        return None
    plan = json.loads((output / "plan.json").read_text())
    saved = json.loads((output / "settings.json").read_text())
    config = json.loads((ROOT / "config.json").read_text())
    result = json.loads(result_path.read_text())
    settings = saved["settings"]
    if plan["run_id"] != run_id or any(saved[key] != config[key] for key in ("model", "revision")):
        raise ValueError("Saved training belongs to a different run or model")
    if settings["gradient_accumulation_steps"] != gradient_accumulation:
        raise ValueError("Saved training accumulation differs; use a new run")
    examples = saved["train_examples"]
    per_epoch = math.ceil(
        math.ceil(examples / settings["per_device_train_batch_size"]) / gradient_accumulation
    )
    if (
        result["optimizer_steps"] != per_epoch * plan["epochs"]
        or result["train"]["epoch"] != plan["epochs"]
    ):
        raise ValueError("Saved training result is incomplete")
    for epoch in range(1, plan["epochs"] + 1):
        checkpoint = output / f"checkpoint-{per_epoch * epoch}"
        if not all(
            (checkpoint / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_config.json")
        ):
            raise ValueError("Saved training checkpoint is incomplete")
    return result


def call_state(call):
    """Poll the result; Modal's best-effort call graph is not a lifecycle signal."""
    try:
        call.get(timeout=0)
    except modal.exception.TimeoutError as exc:
        if type(exc) is modal.exception.TimeoutError:
            return "PENDING"
        if isinstance(exc, modal.exception.FunctionTimeoutError):
            return "FAILURE"
        raise  # Expired results require inspection, not duplicate paid work.
    except (modal.exception.RemoteError, modal.exception.ExecutionError,
            modal.exception.InternalFailure):
        return "FAILURE"
    except (modal.exception.Error, ConnectionError):
        raise  # Authentication/transport failures do not prove the worker stopped.
    except Exception:
        return "FAILURE"  # The remote function returned an application exception.
    return "SUCCESS"


def record_stage(output, stage, **updates):
    from recordings import save
    results.reload()
    path = output / "orchestration.json"
    journal = json.loads(path.read_text())
    journal[stage].update(updates)
    save(path, journal)
    results.commit()


def finish_stage(output, stage, call):
    try:
        value = call.get()
    except Exception:
        if call_state(call) in {"FAILURE", "INIT_FAILURE", "TERMINATED", "TIMEOUT"}:
            record_stage(output, stage, state="failed")
        raise
    record_stage(output, stage, state="done", result=value)
    return value


def durable_stage(output, stage, function, *args, wait=True, **kwargs):
    """Persist child call IDs so coordinator preemption cannot replay a stage."""
    from recordings import save

    results.reload()
    path = output / "orchestration.json"
    journal = json.loads(path.read_text()) if path.exists() else {}
    entry = journal.get(stage)
    if (
        stage == "collection"
        and entry
        and entry.get("state") == "done"
        and (entry.get("result") or {}).get("state") == "waiting_accounts"
    ):
        # The previous call exited cleanly; a new launch may have new/refilled logins.
        entry = None
    request = {"args": list(args), "kwargs": kwargs}
    if entry and entry["request"] != request:
        raise ValueError(f"Saved {stage} arguments differ; use a new run")
    if entry and entry["state"] == "done":
        return entry["result"] if wait else modal.FunctionCall.from_id(entry["call_id"])
    if entry:
        if not entry.get("call_id"):
            # A crash between submission and persisting the call ID has an
            # ambiguous outcome. Never create duplicate paid work automatically.
            raise RuntimeError(
                f"{stage} submission was interrupted; "
                "inspect the existing Modal call before retrying"
            )
        call = modal.FunctionCall.from_id(entry["call_id"])
        if call_state(call) in {"FAILURE", "INIT_FAILURE", "TERMINATED", "TIMEOUT"}:
            entry = None
    if entry is None:
        previous = journal.get(stage, {})
        entry = {"state": "launching", "request": request,
                 "previous_call_ids": previous.get("previous_call_ids", []) +
                 ([previous["call_id"]] if previous.get("call_id") else [])}
        journal[stage] = entry
        save(path, journal)
        results.commit()
        call = function.spawn(*args, **kwargs)
        entry.update(state="running", call_id=call.object_id)
        save(path, journal)
        results.commit()
    if not wait:
        return call
    return finish_stage(output, stage, call)


def publish_collection(plan, output):
    """Publish only this run's prompts and verified teacher artifacts."""
    from recordings import public_recording, save

    cfg = json.loads((ROOT / "config.json").read_text())
    protocol = {
        "promptSet": RUN_NAME + "-validation",
        "promptIds": [t["task_id"] for t in plan["validation_tasks"]],
        "split": "validation",
        "scorer": "Xiaomi " + plan["scorer_revision"][:12],
        "protocol": "greedy-8192-vllm-native-fp8-lora-v3",
        "attentionBackend": "auto",
        "model": cfg["model"],
        "revision": cfg["revision"],
    }
    replay = {
        **protocol,
        "promptSet": RUN_NAME + "-training-replay",
        "promptIds": [t["task_id"] for t in plan["train_tasks"]],
        "split": "train",
    }
    public = Path("/runs/public/runs") / RUN_ID
    target = public / "evaluation.json"
    manifest = (
        json.loads(target.read_text())
        if target.exists()
        else {
            "runId": RUN_ID,
            "reference": protocol,
            "teacher": None,
            "checkpoints": [],
            "recordings": [],
            "recordingStates": [],
            "activity": [],
            "trainingPromptIds": replay["promptIds"],
            "trainingRecordings": [],
            "trainingStates": [],
            "prompts": [
                {
                    "id": t["task_id"],
                    "title": t["task_id"],
                    "prompt": "\n".join(m["content"] for m in t["prompt"]),
                }
                for t in plan["validation_tasks"] + plan["train_tasks"]
            ],
        }
    )
    manifest.setdefault("verifierScores", [])
    rejected_path = output / "data/rejected.json"
    rejected = json.loads(rejected_path.read_text()) if rejected_path.exists() else []
    for split in ("train", "validation") if plan.get("teacher_validation", False) else ("train",):
        saved = output / "data" / (split + ".jsonl")
        rows = (
            {r["task_id"]: r for r in (json.loads(line) for line in saved.read_text().splitlines())}
            if saved.exists()
            else {}
        )
        failures = {r["task_id"]: r for r in rejected if r["split"] == split}
        key = "trainingRecordings" if split == "train" else "recordings"
        for task in plan["train_tasks" if split == "train" else "validation_tasks"]:
            identity = task["task_id"]
            if identity not in rows and identity not in failures:
                continue
            accepted = identity in rows
            if any(
                r["promptId"] == identity and r["split"] == split and r["version"] == "teacher"
                for r in manifest["verifierScores"]
            ):
                continue
            answer_path = (
                output
                / "teacher"
                / split
                / identity
                / str(failures.get(identity, {}).get("attempt", 0))
                / "answer.txt"
            )
            text = (
                rows[identity]["completion"][0]["content"]
                if accepted
                else (answer_path.read_text() if answer_path.exists() else "")
            )
            score = failures.get(identity, {"reward": rows.get(identity, {}).get("reward", 0)})
            if text and not any(r["id"] == identity for r in manifest[key]):
                try:
                    recording, rendered_score = public_recording(
                        task,
                        text,
                        "teacher",
                        RUN_ID + "/native-fp8",
                        public / "native-fp8" / split / identity / "teacher",
                        split,
                        replay if split == "train" else protocol,
                    )
                    if accepted:
                        score = rendered_score
                    if recording:
                        manifest[key].append(recording)
                    elif accepted:
                        score = {**score, "reason": "Teacher playback unavailable"}
                except Exception as exc:
                    score = {**score, "reason": f"Teacher playback failed ({type(exc).__name__})"}
            score_row = {
                "promptId": identity,
                "split": split,
                "version": "teacher",
                "reward": score.get("reward", 0),
                "accepted": accepted,
                "reason": score.get("reason"),
            }
            manifest["verifierScores"] = [
                r
                for r in manifest["verifierScores"]
                if not (
                    r["promptId"] == identity and r["split"] == split and r["version"] == "teacher"
                )
            ] + [score_row]
    manifest["activity"] = status()["activity"]
    save(target, manifest)


def collect():

    from recordings import now, save
    from reward import check_scorer
    from teacher import collect_pending, verify

    check_scorer()
    output = directory()
    output.mkdir(exist_ok=True)
    plan = json.loads(Path("/seed/plan.json").read_text())
    if plan.get("run_name") != RUN_NAME or plan.get("run_id") != RUN_ID:
        raise ValueError("Run identifiers must match the prepared plan")
    if (output / "plan.json").exists() and json.loads((output / "plan.json").read_text()) != plan:
        raise ValueError("Saved run plan differs; refusing to mix collections")
    save(output / "plan.json", plan)
    config = json.loads((ROOT / "config.json").read_text())
    initial = {
        "runId": RUN_ID,
        "runName": RUN_NAME,
        "model": config["model"],
        "createdAt": now(),
        "state": "collecting",
        "step": None,
        "totalSteps": optimizer_steps_per_epoch(len(plan["train_tasks"]), BATCH_SIZE, GRAD_ACC) * plan.get("epochs", 3),
        "trainExamples": len(plan["train_tasks"]),
        "validationExamples": len(plan["validation_tasks"]),
        "acceptedTrain": 0,
        "acceptedValidation": 0,
        "teacherValidationExamples": len(plan["validation_tasks"])
        if plan.get("teacher_validation", False)
        else 0,
        "history": [],
        "events": [],
        "activity": [],
        "lastObservationAt": None,
    }
    fresh = not (output / "status.json").exists()
    if fresh:
        save(output / "status.json", initial)
        from costs import reset_run_costs

        reset_run_costs(output, RUN_ID)
    if plan.get("teacher_source_run"):
        import hashlib

        source_name = plan["teacher_source_run"]
        if (
            not source_name
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for c in source_name
            )
            or source_name == RUN_NAME
        ):
            raise ValueError("Invalid teacher source run")
        source = output.parent / source_name / "data/train.jsonl"
        contents = source.read_bytes()
        if hashlib.sha256(contents).hexdigest() != plan["teacher_source_sha256"]:
            raise ValueError("Source teacher dataset changed")
        rows = [json.loads(line) for line in contents.splitlines()]
        tasks = {t["task_id"]: t for t in plan["train_tasks"]}
        if len(rows) != len(tasks) or {r["task_id"] for r in rows} != set(tasks):
            raise ValueError("Reused teachers must exactly match the training plan")
        for row in rows:
            task = tasks[row["task_id"]]
            if row["prompt"] != task["prompt"]:
                raise ValueError("Reused teacher prompt differs")
            _, score = verify(task, row["completion"][0]["content"])
            if not score["accepted"]:
                raise ValueError("Reused teacher failed verification: " + row["task_id"])
        target = output / "data/train.jsonl"
        target.parent.mkdir(exist_ok=True)
        if target.exists() and target.read_bytes() != contents:
            raise ValueError("Existing training data differs from reused teachers")
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(contents)
        temporary.replace(target)
        publish(
            "Reused and reverified teacher examples from " + source_name + ".",
            acceptedTrain=len(rows),
        )
    publish_collection(plan, output)
    publish(
        "Using saved teacher references for fresh student training."
        if plan.get("teacher_source_run")
        else "Collecting fresh teacher references."
        if fresh
        else "Collecting teacher references; resuming from saved verified answers.",
        state="collecting",
        totalSteps=optimizer_steps_per_epoch(len(plan["train_tasks"]), BATCH_SIZE, GRAD_ACC) * plan.get("epochs", 3),
        batchSize=BATCH_SIZE,
        gradientAccumulation=GRAD_ACC,
    )
    rejected_path = output / "data/rejected.json"
    rejected = json.loads(rejected_path.read_text()) if rejected_path.exists() else []
    for split in ("train", "validation") if plan.get("teacher_validation", False) else ("train",):
        data = output / "data"
        data.mkdir(exist_ok=True)
        saved = data / (split + ".jsonl")
        rows = (
            [json.loads(line) for line in saved.read_text().splitlines()] if saved.exists() else []
        )
        tasks = plan["train_tasks" if split == "train" else "validation_tasks"]
        completed = {r["task_id"] for r in rows} | {
            r["task_id"] for r in rejected if r["split"] == split
        }
        pending = [task for task in tasks if task["task_id"] not in completed]
        order = {task["task_id"]: i for i, task in enumerate(tasks)}
        concurrency = plan.get("teacher_concurrency", 10)

        def progress(message, split=split, rows=rows, tasks=tasks, concurrency=concurrency):
            publish_collection(plan, output)
            rejected_count = sum(r["split"] == split for r in rejected)
            publish(
                message,
                teacherConcurrency=concurrency,
                **{
                    "accepted" + split.title(): len(rows),
                    "rejected" + split.title(): rejected_count,
                    "pending" + split.title(): len(tasks) - len(rows) - rejected_count,
                },
            )

        progress(f"{split.title()} collection: {len(pending)} pending.")
        from codex_accounts import AccountPool

        accounts = AccountPool(commit=lambda name: auth_volumes[name].commit())
        try:
            outcomes = collect_pending(
                pending, output, split, plan.get("teacher_attempts", 1), concurrency, accounts
            )
            for row, failure in outcomes:
                if failure:
                    rejected.append(failure)
                    save(rejected_path, rejected)
                    message = (
                        f"Teacher {failure['task_id']} skipped: {failure['reason']}. Continuing."
                    )
                else:
                    rows.append(row)
                    rows.sort(key=lambda r: order[r["task_id"]])
                    temporary = saved.with_suffix(".tmp")
                    temporary.write_text(
                        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
                    )
                    temporary.replace(saved)
                    message = f"{split.title()} teachers accepted: {len(rows)}/{len(tasks)}."
                progress(message)
        finally:
            outcomes.close()
            accounts.close()
    accepted_counts = {
        split: len((output / "data" / (split + ".jsonl")).read_text().splitlines())
        if (output / "data" / (split + ".jsonl")).exists()
        else 0
        for split in ("train", "validation")
    }
    if not accepted_counts["train"] or (
        plan.get("teacher_validation", False) and not accepted_counts["validation"]
    ):
        raise RuntimeError("Need accepted teacher answers for the configured training protocol")
    publish(
        f"Teacher collection complete; {accepted_counts['train']} accepted training answers, "
        f"{len(rejected)} rejected answers saved. Launching student SFT.",
        state="queued",
        totalSteps=optimizer_steps_per_epoch(accepted_counts["train"], BATCH_SIZE, GRAD_ACC) * plan.get("epochs", 3),
    )


@app.function(
    image=collection_image,
    cpu=2,
    memory=4096,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={
        "/runs": results,
        **{f"/auth/{name}": volume for name, volume in auth_volumes.items()},
    },
)
def collect_teacher():
    try:
        collect()
        return {"state": "complete"}
    except AccountsUnavailable as exc:
        publish(
            str(exc) + ". Add or refill accounts, then resume this run.", state="waiting_accounts"
        )
        return {"state": "waiting_accounts"}
    except Exception as exc:
        if (directory() / "status.json").exists():
            publish("Teacher collection failed: " + str(exc), state="failed")
        raise


@app.function(
    image=tokenizer_image, cpu=2, memory=4096, timeout=1800,
    volumes={"/runs": results, "/cache": cache},
)
def prepare_training(run_name: str):
    from transformers import AutoTokenizer
    from data import filter_training_rows
    from recordings import save
    results.reload()
    output = Path("/runs") / run_name
    config = json.loads((ROOT / "config.json").read_text())
    plan = json.loads((output / "plan.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    report = {}
    for split in ("train", "validation") if plan.get("teacher_validation") else ("train",):
        source = output / "data" / f"{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        kept, excluded = filter_training_rows(rows, tokenizer,
            config["sft"]["trainer"]["max_length"], config.get("chat_template_kwargs", {}))
        report[split] = {"kept": len(kept), "excluded": excluded}
        target = output / "data" / f"{split}-sft.jsonl"
        temporary = target.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
        temporary.replace(target)
    save(output / "training-eligibility.json", report)
    results.commit()
    cache.commit()
    if any(not row["kept"] for row in report.values()):
        raise ValueError("No eligible examples remain after student context filtering")
    count = report["train"]["kept"]
    publish(f"Student preflight: {count} eligible teachers; "
            f"{len(report['train']['excluded'])} overlength answers excluded.",
            run_name=run_name, state="queued", trainingExamples=count,
            totalSteps=optimizer_steps_per_epoch(count, BATCH_SIZE, GRAD_ACC) * plan["epochs"])
    return report


@app.function(
    image=training_image,
    gpu="H100",
    cpu=4,
    memory=65536,
    timeout=28800,
    retries=0,
    max_containers=1,
    volumes={"/runs": results, "/cache": cache},
    secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
)
@track_gpu("training")
def train_student(run_name: str, run_id: str, gradient_accumulation: int = GRAD_ACC):
    global RUN_NAME, RUN_ID
    RUN_NAME, RUN_ID = run_name, run_id
    results.reload()
    finished = completed_training(directory(), run_id, gradient_accumulation)
    if finished is not None:
        return finished
    import torch
    from recordings import now, save
    from sft import build_trainer
    from data import training_data_path
    from transformers import TrainerCallback

    import wandb

    results.reload()
    output = directory()
    plan = json.loads((output / "plan.json").read_text())
    examples = len(training_data_path(output / "data").read_text().splitlines())
    validation_examples = len(plan["validation_tasks"])
    epochs = plan["epochs"]
    steps_per_epoch = optimizer_steps_per_epoch(examples, BATCH_SIZE, gradient_accumulation)
    total_steps = steps_per_epoch * epochs
    cfg = json.loads((ROOT / "config.json").read_text())
    if (
        epochs != cfg["sft"]["trainer"]["num_train_epochs"]
        or gradient_accumulation != cfg["sft"]["trainer"]["gradient_accumulation_steps"]
    ):
        raise ValueError("Run plan and training config differ")
    publish_collection(plan, output)
    publish(
        f"Published {examples} accepted training teachers; "
        f"{validation_examples} held-out prompts are reserved for student evaluation.",
        state="loading",
        totalSteps=total_steps,
        trainingExamples=examples,
    )

    class Progress(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = output / f"checkpoint-{state.global_step}"
            required = (
                "adapter_model.safetensors",
                "adapter_config.json",
                "trainer_state.json",
                "optimizer.pt",
                "scheduler.pt",
            )
            if not all((checkpoint / name).is_file() for name in required):
                raise RuntimeError("Epoch checkpoint is incomplete")
            if plan.get("training_reward_scope") == "optimizer_batch":
                ids = trainer.update_prompt_ids.copy()
                if (
                    not ids
                    or len(ids)
                    > args.per_device_train_batch_size * args.gradient_accumulation_steps
                    or len(set(ids)) != len(ids)
                ):
                    raise RuntimeError("Incomplete optimizer update prompt identities")
                save(
                    checkpoint / "update-prompts.json",
                    {"step": state.global_step, "promptIds": ids},
                )
            results.commit()
            epoch = int(round(state.epoch))
            boundary = abs(state.epoch - epoch) < 1e-6
            if boundary:
                path = output / "epoch-checkpoints.json"
                entries = json.loads(path.read_text()) if path.exists() else []
                entries = [row for row in entries if row["epoch"] != epoch] + [
                    {"id": f"step-{state.global_step}", "step": state.global_step, "epoch": epoch}
                ]
                save(path, sorted(entries, key=lambda row: row["epoch"]))
            save(checkpoint / "ready.json", {"step": state.global_step, "epoch": state.epoch})
            results.commit()
            if plan.get("training_reward_scope") == "optimizer_batch":
                trainer.update_prompt_ids.clear()

        def on_train_begin(self, args, state, control, **kwargs):
            self.previous_tokens = 0
            self.step_started = time.monotonic()

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_started = time.monotonic()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs and "num_tokens" in logs:
                elapsed = time.monotonic() - self.step_started
                token_delta = logs["num_tokens"] - self.previous_tokens
                if elapsed > 0 and token_delta >= 0:
                    logs["tokens_per_second"] = token_delta / elapsed
                    run.log(
                        {
                            "train/tokens_per_second": logs["tokens_per_second"],
                            "train/global_step": state.global_step,
                        }
                    )
                self.previous_tokens = logs["num_tokens"]
            history = status()["history"]
            record = {
                ("eval/" + k[5:] if k.startswith("eval_") else "train/" + k): v
                for k, v in (logs or {}).items()
                if isinstance(v, (int, float)) and math.isfinite(v)
            }
            record.update({"train/global_step": state.global_step, "_timestamp": time.time()})
            history.append(record)
            events = status()["events"]
            import subprocess

            measurement = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                .stdout.strip()
                .splitlines()[0]
                .split(",")
            )
            events.append(
                {
                    "_timestamp": time.time(),
                    "system/gpu.0.gpu": float(measurement[0]),
                    "system/gpu.0.memoryAllocatedBytes": float(measurement[1]) * 1024**2,
                }
            )
            publish(
                f"Optimizer step {state.global_step}/{total_steps}"
                + (f" · loss {logs['loss']:.4f}" if logs and "loss" in logs else "")
                + ".",
                state="training",
                step=state.global_step,
                history=history,
                events=events,
                lastObservationAt=now(),
            )

    try:
        with wandb.init(
            project=cfg["wandb_project"],
            name=RUN_NAME,
            id=RUN_ID,
            resume="allow",
            dir=str(output),
            config={
                "model": cfg["model"],
                "revision": cfg["revision"],
                "teacher": {"model": "gpt-6.1-sol", "reasoning_effort": "low"},
                "purpose": "sft-pilot",
                "train_examples": examples,
                "epochs": epochs,
                "completion_only_loss": True,
                "validation_generation_examples": validation_examples,
                "selection_seed": plan.get("selection_seed"),
                "training_reward_scope": plan.get("training_reward_scope", "fixed_training_set"),
                "evaluation_frequency": plan.get("evaluation_frequency", "epoch"),
            },
        ) as run:
            trainer, settings = build_trainer(
                output / "data",
                output,
                RUN_NAME,
                examples=examples,
                every_step=plan.get("evaluation_frequency") == "optimizer",
                teacher_validation=plan.get("teacher_validation", False),
                track_update_prompts=plan.get("training_reward_scope") == "optimizer_batch",
            )
            trainer.add_callback(Progress())
            run.config.update(settings, allow_val_change=True)
            save(output / "settings.json", settings)
            results.commit()
            publish(
                f"Student SFT started: {epochs} epochs, batch {BATCH_SIZE}, "
                f"accumulation {gradient_accumulation}, {total_steps} optimizer updates.",
                state="training",
                step=0,
            )
            ready = sorted(
                (p for p in output.glob("checkpoint-*") if (p / "ready.json").exists()),
                key=lambda p: int(p.name.split("-")[-1]),
            )
            torch.cuda.reset_peak_memory_stats()
            trained = trainer.train(resume_from_checkpoint=str(ready[-1]) if ready else None)
            trainer.save_model(str(output / "adapter"))
            trainer.save_state()
            norms = [r["grad_norm"] for r in trainer.state.log_history if "grad_norm" in r]
            if (
                trainer.state.global_step != total_steps
                or len(norms) != total_steps
                or not all(math.isfinite(n) and n > 0 for n in norms)
            ):
                raise RuntimeError(f"Expected {total_steps} finite nonzero training updates")
            metrics = {
                "train": trained.metrics,
                "evaluation": trainer.evaluate() if plan.get("teacher_validation", False) else None,
                "optimizer_steps": total_steps,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                "wandb_url": run.url,
            }
            save(output / "result.json", metrics)
            run.summary.update(metrics)
            publish(
                "Training complete. Saved steps "
                f"{[steps_per_epoch * i for i in range(1, epochs + 1)]}; "
                "preparing held-out student evaluation.",
                state="evaluating",
                step=total_steps,
            )
            return metrics
    except Exception as exc:
        publish(
            "Student training failed (" + type(exc).__name__ + "); see Modal logs.", state="failed"
        )
        raise
    finally:
        cache.commit()
        results.commit()


def supervise_workers(output, training, evaluation):
    """Keep training if evaluation fails; stop evaluation if training fails."""
    from recordings import now, save
    pending = {"training": training, "evaluation": evaluation}
    values, errors = {}, {}
    while pending:
        for stage, call in list(pending.items()):
            if call_state(call) == "PENDING":
                continue
            try:
                values[stage] = finish_stage(output, stage, call)
            except Exception as exc:
                errors[stage] = exc
                if stage == "training":
                    if "evaluation" in pending:
                        evaluation.cancel(terminate_containers=True)
                    publish("Training failed; stopping inference. Relaunch to resume saved work.",
                            state="failed")
                    raise
                save(output / "evaluation-status.json", {
                    "state": "failed", "message": "Inference failed; training continues. Relaunch to resume evaluation.",
                    "updatedAt": now(),
                })
                results.commit()
            del pending[stage]
        if pending:
            time.sleep(10)
    if errors:
        raise errors["evaluation"]
    return values["evaluation"]


@app.function(
    image=coordinator_image,
    cpu=0.25,
    memory=256,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/runs": results},
)
def coordinate():
    try:
        return coordinate_run()
    except Exception:
        results.reload()
        if (directory() / "status.json").exists():
            publish("Run interrupted; saved work is preserved. Relaunch to resume.", state="failed")
        raise


def coordinate_run():
    results.reload()
    output = directory()
    finished = completed_training(output, RUN_ID, GRAD_ACC)
    if finished is None:
        collection = durable_stage(output, "collection", collect_teacher)
        if collection and collection.get("state") == "waiting_accounts":
            return collection
        prepare = modal.Function.from_name("mimo-music-inference", "prepare_model")
        durable_stage(output, "preparation", prepare)
        durable_stage(output, "eligibility", prepare_training, RUN_NAME)
        plan = json.loads((output / "plan.json").read_text())
        if plan.get("stream_evaluation", plan.get("evaluate_every_step", False)):
            publish("Starting or resuming student training.", state="queued")
            training = durable_stage(
                output, "training", train_student, RUN_NAME, RUN_ID, GRAD_ACC, wait=False
            )
            # Wait on the inexpensive coordinator, not an idle inference H100.
            deadline = time.monotonic() + 7200
            while True:
                if call_state(training) != "PENDING":
                    finish_stage(output, "training", training)
                results.reload()
                if (output / "settings.json").is_file():
                    break
                if time.monotonic() >= deadline:
                    training.cancel(terminate_containers=True)
                    raise TimeoutError("Training settings were not published; worker stopped")
                time.sleep(5)
            inference = modal.Function.from_name("mimo-music-inference", "run")
            evaluation = durable_stage(
                output,
                "evaluation",
                inference,
                evaluate=True,
                run_name=RUN_NAME,
                concurrency=64,
                wait=False,
            )
            value = supervise_workers(output, training, evaluation)
            results.reload()
            publish("Training and epoch checkpoint evaluation completed.", state="finished")
            return value
        durable_stage(output, "training", train_student, RUN_NAME, RUN_ID, GRAD_ACC)
    publish("Training is saved; starting or resuming evaluation.", state="evaluating")
    inference = modal.Function.from_name("mimo-music-inference", "run")
    value = durable_stage(
        output, "evaluation", inference, evaluate=True, run_name=RUN_NAME, concurrency=64
    )
    results.reload()
    publish("Training and evaluation completed.", state="finished")
    return value


@app.local_entrypoint()
def main():
    if not RUN_NAME or not RUN_ID:
        raise ValueError("Prepare a run first, then set MUSIC_RUN_NAME and MUSIC_RUN_ID")
    call = coordinate.spawn()
    print("Detached coordinator: " + call.object_id, flush=True)
    print(
        "W&B run ID: " + RUN_ID,
        flush=True,
    )
