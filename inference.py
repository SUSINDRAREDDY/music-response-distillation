"""Continuously scheduled vLLM evaluation; training stays in TRL."""

import json
import math
import os
import time
from itertools import groupby
from pathlib import Path

import modal
from recordings import io_lock, track_gpu

ROOT = Path(__file__).resolve().parent
app = modal.App("mimo-music-inference")
cache = modal.Volume.from_name("mimo-music-model-cache")
results = modal.Volume.from_name("mimo-music-runs")
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.31.0")
    .entrypoint([])
    .apt_install("build-essential")
    .run_commands("test -e /usr/local/bin/python || ln -s /usr/bin/python3 /usr/local/bin/python")
    .pip_install("wandb==0.25.0", "nvidia-nccl-cu13==2.29.7")
    .run_commands(
        "python -m pip uninstall -y opentelemetry-exporter-otlp-common opentelemetry-exporter-http-transport"
    )
    .run_commands("python -m pip check")
    .add_local_file(ROOT / "setup_scorer.py", "/root/setup_scorer.py", copy=True)
    .run_commands("python /root/setup_scorer.py")
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TRITON_CACHE_DIR": "/cache/vllm-0.31-triton",
            "VLLM_CACHE_ROOT": "/cache/vllm-0.31",
            "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "ABC2MIDI_BIN": "/root/tools/abc2midi",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
        }
    )
)
prepare_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("modal==1.6.1")
    .add_local_file(ROOT / "setup_scorer.py", "/root/setup_scorer.py")
)
for name in (
    "inference.py",
    "inference_engine.py",
    "recordings.py",
    "teacher.py",
    "codex_accounts.py",
    "data.py",
    "reward.py",
    "config.json",
    "pipeline.py",
):
    image = image.add_local_file(ROOT / name, f"/root/{name}")
    prepare_image = prepare_image.add_local_file(ROOT / name, f"/root/{name}")
image = image.add_local_dir(ROOT / "vendor", "/root/vendor", ignore=["**/__pycache__/**"])
prepare_image = prepare_image.add_local_dir(
    ROOT / "vendor", "/root/vendor", ignore=["**/__pycache__/**"]
)


def commit_results():
    with io_lock:
        results.commit()


def reload_results():
    with io_lock:
        results.reload()


def throughput_sample(current, previous, sampled_at):
    """Windowed throughput; the completed-batch average is a separate metric."""
    value = dict(current)
    tokens = value.get("outputTokens", 0)
    batch = value.get("batchId")
    rate = 0.0
    if value.get("state") == "generating":
        if previous and previous[0] == batch:
            rate = max(0, tokens - previous[1]) / max(sampled_at - previous[2], .001)
        else:
            rate = tokens / max(value.get("elapsedSeconds", 0), .001)
    value["outputTokensPerSecond"] = rate
    return value, (batch, tokens, sampled_at)


def validate_outputs(outputs, expected):
    if len(outputs) != expected or any(
        len(row.outputs) != 1 or not row.outputs[0].token_ids for row in outputs
    ):
        raise RuntimeError("vLLM returned an incomplete batch")


def upsert_score(manifest, score):
    keys = ("promptId", "split", "version")
    manifest["verifierScores"] = [
        row
        for row in manifest.get("verifierScores", [])
        if any(row[key] != score[key] for key in keys)
    ] + [score]


def evaluation_schedule(
    steps,
    train_tasks,
    validation_tasks,
    streaming=False,
    epoch_steps=None,
    every_optimizer=False,
    batch_scope=False,
    baseline=False,
):
    """Score both fixed prompt sets at completed epoch checkpoints."""
    selected = steps if every_optimizer else epoch_steps or steps
    jobs = ([(0, "train", train_tasks), (0, "validation", validation_tasks)]
            if baseline else [] if streaming else [(0, "validation", validation_tasks)])
    for step in selected:
        jobs.append((step, "train", train_tasks))
        if not every_optimizer or step in (epoch_steps or steps):
            if batch_scope:
                jobs.append((step, "gallery", train_tasks))
            jobs.append((step, "validation", validation_tasks))
    return jobs


def optimizer_batch_tasks(checkpoint, tasks):
    update = json.loads((checkpoint / "update-prompts.json").read_text())
    ids = update.get("promptIds", [])
    if (
        update.get("step") != int(checkpoint.name.split("-")[-1])
        or not ids
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Invalid optimizer update prompt identities")
    by_id = {task["task_id"]: task for task in tasks}
    return [by_id[identity] for identity in ids]


def evaluation_log_directory(run_id):
    import tempfile

    return tempfile.mkdtemp(prefix=f"music-{run_id}-eval-")


def checkpoint_mean(scores, tasks, split, version):
    ids = {task["task_id"] for task in tasks}
    rows = [
        row
        for row in scores
        if row["split"] == split and row["version"] == version and row["promptId"] in ids
    ]
    if len(rows) != len(ids) or {row["promptId"] for row in rows} != ids:
        raise RuntimeError(f"Incomplete {split} scores at {version}")
    return sum(row["reward"] for row in rows) / len(ids)


def pending_tasks(manifest, tasks, split, version):
    scored = {
        row["promptId"]
        for row in manifest.get("verifierScores", [])
        if row["split"] == split and row["version"] == version
    }
    return [task for task in tasks if task["task_id"] not in scored]


def wait_for_file(path, output, timeout=7200):
    deadline = time.monotonic() + timeout
    while True:
        reload_results()
        status_path = output / "status.json"
        if (
            status_path.exists()
            and not (output / "result.json").exists()
            and json.loads(status_path.read_text()).get("state") == "failed"
        ):
            raise RuntimeError("Training failed; stopping checkpoint evaluation")
        if path.is_file():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path.name}")
        time.sleep(10)


def text_model_path(config):
    return Path("/cache/text-models") / config["model"].replace("/", "--") / config["revision"]


def validate_adapter(path, config):
    """Reject a checkpoint from a different base before creating the engine."""
    adapter = json.loads((path / "adapter_config.json").read_text())
    if adapter.get("base_model_name_or_path") != config["model"]:
        raise ValueError(f"Adapter base model differs: {path}")
    if adapter.get("revision") not in (None, config["revision"]):
        raise ValueError(f"Adapter revision differs: {path}")
    if adapter.get("r") != config["sft"]["lora"]["r"]:
        raise ValueError(f"Adapter rank differs: {path}")
    if not (path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(path / "adapter_model.safetensors")


def zero_rate_warmup_checkpoint(path):
    """Only an initial zero-rate update may retain the identity LoRA."""
    from safetensors.torch import load_file

    state = json.loads((path / "trainer_state.json").read_text())
    first = next((row for row in state.get("log_history", []) if row.get("step") == 1), {})
    if state.get("global_step") != 1 or first.get("learning_rate") != 0:
        return False
    tensors = load_file(path / "adapter_model.safetensors")
    matrices = [tensor for key, tensor in tensors.items() if ".lora_B." in key]
    return bool(matrices) and all(tensor.count_nonzero().item() == 0 for tensor in matrices)


def check_adapter(engine, request, prompt, base=None, allow_identity=False):
    """Public-API behavioral check; no worker RPC or model monkey patches."""
    from vllm import SamplingParams

    probe = SamplingParams(temperature=0, max_tokens=32, logprobs=1, seed=42)
    if base is None:
        base = engine.generate([prompt], probe, use_tqdm=False)[0].outputs[0]
    adapted = engine.generate([prompt], probe, lora_request=request, use_tqdm=False)[0].outputs[0]
    changed = {
        "changed_tokens": base.token_ids != adapted.token_ids,
        "changed_logprobs": base.logprobs != adapted.logprobs,
    }
    if not any(changed.values()) and not allow_identity:
        raise RuntimeError(f"Adapter {request.lora_name} has no observable effect")
    if allow_identity:
        changed["identity_reason"] = (
            "Initial warmup update used zero learning rate; all LoRA B matrices are zero."
        )
    return changed


@app.function(
    image=image, cpu=8, memory=131072, timeout=7200, max_containers=1, volumes={"/cache": cache}
)
def export_model():
    """One-time CPU export using stock Transformers; adapters stay untouched."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from recordings import save

    cache.reload()
    config = json.loads((ROOT / "config.json").read_text())
    destination = text_model_path(config)
    marker = destination / "export.json"
    if marker.exists():
        return str(destination)
    temporary = destination.with_name(destination.name + ".partial")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        revision=config["revision"],
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size="4GB")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    tokenizer.save_pretrained(temporary)
    save(
        temporary / "export.json",
        {
            "model": config["model"],
            "revision": config["revision"],
            "architecture": model.config.architectures,
        },
    )
    temporary.rename(destination)
    cache.commit()
    return str(destination)


@app.function(
    image=prepare_image,
    cpu=0.25,
    memory=512,
    timeout=7500,
    max_containers=1,
    volumes={"/cache": cache},
)
def prepare_model():
    """Check the cache before requesting the large export container."""
    cache.reload()
    config = json.loads((ROOT / "config.json").read_text())
    destination = text_model_path(config)
    if (destination / "export.json").is_file():
        return str(destination)
    return export_model.remote()


def main_job(
    evaluate=True,
    attention_backend="auto",
    run_name="",
    concurrency=64,
    structure_guard=None,
    evaluation_frequency=None,
):
    if not run_name:
        raise ValueError("A run name is required")
    if not evaluate:
        raise ValueError("Benchmark mode was removed; use evaluate=True")
    reload_results()
    cache.reload()
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    import wandb
    import vllm
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from inference_engine import ContinuousEngine
    from teacher import requested_bars
    from vllm.lora.request import LoRARequest
    from recordings import now, public_recording, save
    from reward import check_scorer

    torch.set_num_threads(1)
    device_name = torch.cuda.get_device_name(0)
    device_family = "h100"

    output = Path("/runs") / run_name
    plan = json.loads((output / "plan.json").read_text())
    config = json.loads((ROOT / "config.json").read_text())
    if structure_guard is None:
        structure_guard = plan.get("structure_guard", False)
    streaming = plan.get("stream_evaluation", plan.get("evaluate_every_step", False))
    if streaming:
        wait_for_file(output / "settings.json", output)
    trained = json.loads((output / "result.json").read_text()) if not streaming else None
    examples = len(plan["train_tasks"])
    tasks = plan["validation_tasks"]
    if not tasks or not examples:
        raise ValueError("Evaluation requires nonempty validation and training prompts")
    saved = json.loads((output / "settings.json").read_text())
    train_tasks = plan["train_tasks"]
    if plan.get("training_reward_scope") == "fixed_training_set":
        accepted_ids = saved["train_task_ids"]
        by_id = {task["task_id"]: task for task in train_tasks}
        if not accepted_ids or len(set(accepted_ids)) != len(accepted_ids):
            raise ValueError("Require unique accepted training prompt IDs")
        train_tasks = [by_id[identity] for identity in accepted_ids]
    trainer_settings = saved["settings"]
    training_examples = saved.get("train_examples", examples)
    steps_per_epoch = math.ceil(
        math.ceil(training_examples / trainer_settings["per_device_train_batch_size"])
        / trainer_settings["gradient_accumulation_steps"]
    )
    final_step = steps_per_epoch * plan["epochs"]
    epoch_steps = tuple(steps_per_epoch * epoch for epoch in range(1, plan["epochs"] + 1))
    steps = epoch_steps
    frequency = evaluation_frequency or plan.get("evaluation_frequency", "epoch")
    if frequency not in ("epoch", "optimizer"):
        raise ValueError("evaluation_frequency must be epoch or optimizer")
    max_concurrency = 64 if device_family == "h100" else 16
    if not 1 <= concurrency <= max_concurrency:
        raise ValueError(f"concurrency must be between 1 and {max_concurrency} on {device_name}")
    every_optimizer = frequency == "optimizer"
    batch_scope = plan.get("training_reward_scope") == "optimizer_batch"
    epochs_path = output / "epoch-checkpoints.json"
    if epochs_path.exists():
        entries = json.loads(epochs_path.read_text())
        if len(entries) == plan["epochs"]:
            epoch_steps = tuple(
                row["step"] for row in sorted(entries, key=lambda row: row["epoch"])
            )
            steps = epoch_steps
    if every_optimizer:
        steps = tuple(range(1, final_step + 1))
    if trained is not None and (
        trained["optimizer_steps"] != final_step or trained["train"]["epoch"] != plan["epochs"]
    ):
        raise RuntimeError("Require the completed training run")
    for step in () if streaming else steps:
        if not (output / f"checkpoint-{step}" / "adapter_model.safetensors").is_file():
            raise RuntimeError(f"Checkpoint {step} is missing")
    check_scorer()
    saved = json.loads((output / "settings.json").read_text())
    if any(saved[key] != config[key] for key in ("model", "revision")):
        raise ValueError("Training settings do not match the configured base model/revision")
    adapters = output
    for step in () if streaming else steps:
        validate_adapter(adapters / f"checkpoint-{step}", config)
    model_path = text_model_path(config)
    if not (model_path / "export.json").is_file():
        raise RuntimeError("Run the CPU prepare_model function before GPU inference")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def prompts(tasks):
        return [
            tokenizer.apply_chat_template(
                task["prompt"],
                add_generation_prompt=True,
                tokenize=False,
                **config["chat_template_kwargs"],
            )
            for task in tasks
        ]

    if attention_backend not in ("auto", "FLASH_ATTN", "FLASHINFER"):
        raise ValueError("attention_backend must be auto, FLASH_ATTN, or FLASHINFER")
    concurrency = min(concurrency, len(train_tasks) + len(tasks))
    engine_settings = dict(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype="bfloat16",
        quantization="fp8_per_tensor",
        enable_lora=True,
        max_lora_rank=config["sft"]["lora"]["r"],
        max_loras=1,
        max_cpu_loras=3,
        gpu_memory_utilization=0.90,
        max_model_len=12288,
        max_num_seqs=concurrency,
        compilation_config={"max_cudagraph_capture_size": 1 << (concurrency - 1).bit_length()},
        max_num_batched_tokens=4096,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        generation_config="vllm",
        seed=42,
    )
    if attention_backend != "auto":
        engine_settings["attention_config"] = {"backend": attention_backend}

    def progress(message):
        if streaming:
            save(
                output / "evaluation-status.json",
                {"state": "evaluating", "message": message, "updatedAt": now()},
            )
            commit_results()
            print(message, flush=True)
        elif (output / "status.json").exists():
            from pipeline import publish

            publish(message, run_name=run_name, state="evaluating")
        else:
            print(message, flush=True)

    progress("Evaluation container ready; loading vLLM weights and initializing kernels.")
    started = time.monotonic()
    engine = ContinuousEngine(**engine_settings)
    _live_engine["engine"] = engine
    startup_seconds = time.monotonic() - started
    progress(f"vLLM ready in {startup_seconds:.1f}s; verifying checkpoint adapters.")
    checks = {}
    checks_started = time.monotonic()
    probe_prompt = prompts(tasks[:1])[0]
    probe = SamplingParams(temperature=0, max_tokens=32, logprobs=1, seed=42)
    base_probe = engine.generate([probe_prompt], probe, use_tqdm=False)[0].outputs[0]
    for step in () if streaming else steps:
        checks[str(step)] = check_adapter(
            engine,
            LoRARequest(f"step-{step}", step, str(adapters / f"checkpoint-{step}")),
            probe_prompt,
            base=base_probe,
            allow_identity=zero_rate_warmup_checkpoint(adapters / f"checkpoint-{step}"),
        )
    diagnostics = {
        "vllm": str(vllm.__version__),
        "torch": str(torch.__version__),
        "settings": engine_settings,
        "startup_seconds": startup_seconds,
        "gpu": device_name,
        "adapter_checks": checks,
        "adapter_check_seconds": time.monotonic() - checks_started,
        "at": now(),
    }
    save(output / "inference-startup.json", diagnostics)
    save(output / "adapter-check-native-fp8.json", checks)
    commit_results()
    cache.commit()

    public = Path("/runs/public/runs") / plan["run_id"]
    snapshot = public / "evaluation-before-vllm.json"
    if not snapshot.exists():
        save(snapshot, json.loads((public / "evaluation.json").read_text()))
    protocol = {
        "promptSet": plan["run_name"] + "-validation",
        "promptIds": [t["task_id"] for t in tasks],
        "split": "validation",
        "scorer": "Xiaomi " + plan["scorer_revision"][:12],
        "protocol": "greedy-8192-vllm-grouped-fp8-lora-v8",
        "attentionBackend": attention_backend,
        "structureGuard": structure_guard,
        "concurrency": concurrency,
        "evaluationFrequency": frequency,
        "gpu": device_name,
        "model": config["model"],
        "revision": config["revision"],
    }
    if plan.get("baseline_generation"):
        protocol.update(generation=plan["generation"], baselineGeneration=True,
                        trainingRewardScope=plan["training_reward_scope"],
                        replayPromptIds=[t["task_id"] for t in train_tasks])
    replay = {
        **protocol,
        "promptSet": plan["run_name"] + "-training-replay",
        "promptIds": [t["task_id"] for t in train_tasks],
        "split": "train",
    }
    prior = json.loads((public / "evaluation.json").read_text())
    manifest = {
        "runId": plan["run_id"],
        "reference": protocol,
        "teacher": prior.get("teacher"),
        "checkpoints": [],
        "recordings": [],
        "recordingStates": [],
        "activity": json.loads((public / "evaluation.json").read_text()).get("activity", []),
        "trainingPromptIds": [t["task_id"] for t in plan["train_tasks"]],
        "trainingRecordings": [],
        "trainingStates": [],
        "prompts": [
            {
                "id": t["task_id"],
                "title": t["task_id"],
                "prompt": "\n".join(m["content"] for m in t["prompt"]),
            }
            for t in tasks
        ],
    }
    manifest["verifierScores"] = [
        row for row in prior.get("verifierScores", []) if row["version"] == "teacher"
    ]
    manifest["prompts"] = prior.get("prompts", manifest["prompts"])
    artifact_namespace = f"grouped-fp8-v8-{device_family}-c{concurrency}-guard{int(structure_guard)}-{frequency}-{attention_backend.lower()}"
    if plan.get("baseline_generation"):
        artifact_namespace += "-baseline-v6"
    fast = public / artifact_namespace
    manifest["artifactNamespace"] = artifact_namespace
    saved_manifest = fast / "evaluation.json"
    if saved_manifest.exists():
        manifest = json.loads(saved_manifest.read_text())
        if manifest["reference"] != protocol:
            raise RuntimeError("Saved inference protocol differs")

    manifest["plannedCheckpoints"] = [
        {"id": f"step-{step}", "step": step, "epoch": math.ceil(step / steps_per_epoch)}
        for step in epoch_steps
    ]

    manifest["plannedTrainingCheckpoints"] = [
        {"id": f"step-{step}", "step": step, "epoch": math.ceil(step / steps_per_epoch)}
        for step in steps
    ]
    if plan.get("training_reward_scope"):
        manifest["trainingRewardScope"] = plan["training_reward_scope"]
    manifest["replayPromptIds"] = [t["task_id"] for t in train_tasks]
    manifest["baselineGeneration"] = plan.get("baseline_generation", False)

    last_published = 0.0

    def publish(message, step=None, split=None, force=True):
        nonlocal last_published
        if not force and time.monotonic() - last_published < 10:
            return
        last_published = time.monotonic()
        manifest["activity"].append({"at": now(), "message": message})
        save(saved_manifest, manifest)
        save(public / "evaluation.json", manifest)
        if streaming:
            save(
                output / "evaluation-status.json",
                {
                    "state": "finished"
                    if message.startswith("Training and vLLM evaluation completed")
                    else "evaluating",
                    "message": message,
                    "updatedAt": now(),
                    "step": step,
                    "split": split,
                    "readySteps": [
                        entry["step"]
                        for entry in manifest["plannedTrainingCheckpoints"]
                        if (output / f"checkpoint-{entry['step']}" / "ready.json").is_file()
                    ],
                },
            )
            commit_results()
        else:
            commit_results()
        if not streaming and (output / "status.json").exists():
            from pipeline import publish as publish_progress

            publish_progress(
                message,
                run_name=run_name,
                state="finished"
                if message.startswith("Training and vLLM evaluation completed")
                else "evaluating",
            )
        print(message, flush=True)

    def append(row, split):
        recordings = manifest["trainingRecordings" if split == "train" else "recordings"]
        original = next((s for s in recordings if s["id"] == row["id"]), None)
        if original:
            original["compositions"].extend(
                c
                for c in row["compositions"]
                if not any(p["version"] == c["version"] for p in original["compositions"])
            )
        else:
            recordings.append(row)

    if (output / "status.json").exists():
        for split, key in (("train", "trainingRecordings"), ("validation", "recordings")):
            for row in prior.get(key, []):
                teacher = {
                    **row,
                    "compositions": [c for c in row["compositions"] if c["version"] == "teacher"],
                }
                if teacher["compositions"]:
                    append(teacher, split)
    else:
        for task in plan["train_tasks"]:
            text = (
                (public / "native-fp8" if examples == 20 else public)
                / "train"
                / task["task_id"]
                / "teacher"
                / "answer.txt"
            ).read_text()
            row, _ = public_recording(
                task,
                text,
                "teacher",
                plan["run_id"] + "/" + artifact_namespace,
                fast / "train" / task["task_id"] / "teacher",
                "train",
                replay,
            )
            if not row:
                raise RuntimeError("Saved teacher failed rendering")
            append(row, "train")
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    generation = plan["generation"]
    if generation.get("do_sample") is not False:
        raise ValueError("This comparison protocol requires greedy generation")
    sampling = SamplingParams(temperature=0, max_tokens=generation["max_new_tokens"],
                              seed=generation["seed"], stop_token_ids=[eos])
    with wandb.init(
        project=config["wandb_project"],
        name=plan["run_name"] + "-" + artifact_namespace,
        id=plan["run_id"] + "-" + artifact_namespace,
        resume="allow",
        dir=evaluation_log_directory(plan["run_id"]),
    ) as run:
        run.config.update(
            {"inference": {**engine_settings, "generation": generation, "wall_time_limit": None}},
            allow_val_change=True,
        )
        run.summary["inference_startup"] = diagnostics
        run.summary["invalid_evaluation_protocol"] = (
            "greedy-8192-vllm-nf4-bf16-v1: trained adapters were skipped"
        )
        run.summary["native_fp8_adapter_checks"] = checks
        publish(
            "Optimizer checkpoint evaluation is running."
            if every_optimizer
            else "Epoch checkpoint evaluation is running."
            if streaming
            else "Training complete. Batched vLLM evaluation is running."
        )
        manifest.setdefault("trainingCheckpoints", [])
        schedule = evaluation_schedule(
            steps, train_tasks, tasks, streaming, epoch_steps, every_optimizer, batch_scope,
            plan.get("baseline_generation", False)
        )
        for step, grouped_jobs in groupby(schedule, key=lambda job: job[0]):
            jobs = [(job, split_tasks) for _, job, split_tasks in grouped_jobs]
            identity = f"step-{step}"
            if streaming and step > 0:
                _live_engine["phase"] = "waiting"
                wait_for_file(adapters / f"checkpoint-{step}" / "ready.json", output)
                validate_adapter(adapters / f"checkpoint-{step}", config)
                if str(step) not in checks:
                    _live_engine["phase"] = "checking"
                    checks[str(step)] = check_adapter(
                        engine,
                        LoRARequest(
                            f"optimizer-{step}", step, str(adapters / f"checkpoint-{step}")
                        ),
                        probe_prompt,
                        base=base_probe,
                        allow_identity=zero_rate_warmup_checkpoint(adapters / f"checkpoint-{step}"),
                    )
                    save(output / "adapter-check-native-fp8.json", checks)
                    commit_results()
            resolved_jobs = []
            for job, split_tasks in jobs:
                if batch_scope and job == "train":
                    split_tasks = optimizer_batch_tasks(
                        adapters / f"checkpoint-{step}", plan["train_tasks"]
                    )
                    entry = next(row for row in manifest["plannedTrainingCheckpoints"] if row["id"] == identity)
                    entry["promptIds"] = [task["task_id"] for task in split_tasks]
                resolved_jobs.append((job, split_tasks))
            # One native vLLM queue per checkpoint; split tags keep denominators separate.
            pending_by_id = {}
            for job, split_tasks in resolved_jobs:
                split = "train" if job == "gallery" else job
                for task in pending_tasks(manifest, split_tasks, split, identity):
                    pending_by_id[(split, task["task_id"])] = (split, task)
            pending = list(pending_by_id.values())
            lora = (
                None
                if step == 0
                else LoRARequest(f"step-{step}", step, str(adapters / f"checkpoint-{step}"))
            )
            publish(f"Step {step}: generating {len(pending)} training and held-out outputs.", step, "mixed")
            _live_engine["phase"] = "generating"
            engine.telemetry = {}
            started = time.monotonic()
            total_tokens = 0
            completion_seconds = 0.0
            expected_bars = [requested_bars(task) for _, task in pending] if structure_guard else None
            for index, result, metrics in engine.completed(
                prompts([task for _, task in pending]), sampling, lora, expected_bars
            ):
                split, task = pending[index]
                generated = result.outputs[0] if result.outputs else None
                if generated is None or not generated.token_ids:
                    metrics["abort_reason"] = "Generation returned no output tokens"
                directory = fast / split / task["task_id"] / identity
                scoring_started = time.monotonic()
                row, score = public_recording(
                    task,
                    generated.text if generated else "",
                    identity,
                    plan["run_id"] + "/" + artifact_namespace,
                    directory,
                    split,
                    replay if split == "train" else protocol,
                    generation_abort=metrics["abort_reason"],
                )
                metrics.update(
                    scoring_seconds=time.monotonic() - scoring_started,
                    backend="vllm-continuous",
                    published_seconds=time.monotonic() - started,
                )
                upsert_score(
                    manifest,
                    {
                        "promptId": task["task_id"],
                        "split": split,
                        "version": identity,
                        "reward": score.get("reward", 0),
                        "accepted": score["accepted"],
                        "reason": score.get("reason"),
                    },
                )
                save(directory / "generation.json", metrics)
                if row:
                    append(row, split)
                else:
                    key = "trainingStates" if split == "train" else "recordingStates"
                    manifest[key] = [
                        cell
                        for cell in manifest[key]
                        if not (
                            cell["promptId"] == task["task_id"] and cell["checkpointId"] == identity
                        )
                    ]
                    manifest[key].append(
                        {"promptId": task["task_id"], "checkpointId": identity, "status": "failed"}
                    )
                total_tokens += metrics["tokens"]
                completion_seconds = max(completion_seconds, metrics["request_seconds"])
                publish(
                    f"Step {step}: {split} output scored: {task['task_id']} ({metrics['finish_reason']}).",
                    step, "mixed", force=False,
                )
            if total_tokens:
                run.log({
                    "generation/step": step,
                    "generation/tokens_per_second": total_tokens / completion_seconds,
                    "generation/wall_seconds": time.monotonic() - started,
                })
            for job, split_tasks in resolved_jobs:
                split = "train" if job == "gallery" else job
                reward = checkpoint_mean(manifest["verifierScores"], split_tasks, split, identity)
                if job == "gallery":
                    if step == final_step:
                        manifest["trainingReplay"] = {
                            "checkpointId": identity,
                            "sampleCount": len(split_tasks),
                            "reward": reward,
                        }
                    publish(
                        f"Step {step}: all {len(split_tasks)} epoch music results published.",
                        step,
                        "gallery",
                    )
                    continue
                key = "trainingCheckpoints" if split == "train" else "checkpoints"
                checkpoint = {
                    "id": identity,
                    "step": step,
                    "reward": reward,
                    "sampleCount": len(split_tasks),
                    "completedAt": now(),
                }
                if batch_scope and split == "train":
                    checkpoint["promptIds"] = [task["task_id"] for task in split_tasks]
                manifest[key] = [row for row in manifest[key] if row["id"] != identity] + [checkpoint]
                run.log(
                    {
                        f"{'training_results' if split == 'train' else 'evaluation_verified'}/step": step,
                        f"{'training_results' if split == 'train' else 'evaluation_verified'}/reward": reward,
                    }
                )
                if split == "train" and step == final_step and not batch_scope:
                    manifest["trainingReplay"] = {
                        "checkpointId": identity,
                        "sampleCount": len(split_tasks),
                        "reward": reward,
                    }
                publish(
                    f"Step {step}: all {len(split_tasks)} {split} outputs scored; mean reward {reward:.3f}.",
                    step,
                    split,
                )
        # Test prompts are generated only for the final checkpoint, never selected on.
        test_tasks = plan.get("test_tasks", [])
        if test_tasks:
            test_protocol = {**protocol, "promptSet": run_name + "-test", "split": "test",
                             "promptIds": [task["task_id"] for task in test_tasks]}
            test_path = public / "test-evaluation.json"
            test_result = json.loads(test_path.read_text()) if test_path.exists() else {
                "runId": plan["run_id"], "checkpointId": f"step-{final_step}",
                "reference": test_protocol, "scores": [], "recordings": [], "reward": None}
            if test_result["reference"] != test_protocol or test_result["checkpointId"] != f"step-{final_step}":
                raise ValueError("Saved test evaluation protocol differs")
            scored_ids = {row["promptId"] for row in test_result["scores"]}
            pending_test = [task for task in test_tasks if task["task_id"] not in scored_ids]
            publish(f"Final test evaluation: {len(scored_ids)}/{len(test_tasks)} scored.", final_step, "test")
            _live_engine["phase"] = "generating"
            test_lora = LoRARequest(f"step-{final_step}", final_step, str(adapters / f"checkpoint-{final_step}"))
            for index, result, metrics in engine.completed(
                prompts(pending_test), sampling, test_lora,
                [requested_bars(task) for task in pending_test] if structure_guard else None,
            ):
                task = pending_test[index]
                generated = result.outputs[0] if result.outputs else None
                abort = metrics.get("abort_reason") or ("Generation returned no output tokens" if generated is None or not generated.token_ids else None)
                row, score = public_recording(task, generated.text if generated else "", f"step-{final_step}",
                    plan["run_id"] + "/" + artifact_namespace,
                    fast / "test" / task["task_id"] / f"step-{final_step}", "test", test_protocol,
                    generation_abort=abort)
                test_result["scores"].append({"promptId": task["task_id"], "reward": score.get("reward", 0),
                                              "accepted": score["accepted"], "reason": score.get("reason")})
                if row:
                    test_result["recordings"].append(row)
                save(test_path, test_result)
                publish(f"Final test evaluation: {len(test_result['scores'])}/{len(test_tasks)} scored.", final_step, "test")
            if {row["promptId"] for row in test_result["scores"]} != set(test_protocol["promptIds"]) or len(test_result["scores"]) != len(test_tasks):
                raise RuntimeError("Incomplete final test evaluation")
            test_result.update(reward=sum(row["reward"] for row in test_result["scores"]) / len(test_tasks),
                               sampleCount=len(test_tasks), completedAt=now())
            save(test_path, test_result)
            save(output / "test-evaluation.json", test_result)
            commit_results()
            run.summary["test_verified/reward"] = test_result["reward"]
            run.summary["test_verified/sample_count"] = len(test_tasks)
        run.summary["evaluation_verified/training_replay_reward"] = manifest["trainingReplay"][
            "reward"
        ]
        playable = sum(
            any(c["version"] == f"step-{final_step}" for c in row["compositions"])
            for row in manifest["trainingRecordings"]
        )
        publish(
            f"Training and vLLM evaluation completed. {playable}/{len(train_tasks)} student training replays are playable; every output was scored."
        )
    return {"startup": diagnostics, "checkpoints": manifest["checkpoints"]}


_live_engine = {}

def execute_evaluation(
    evaluate: bool = True,
    attention_backend: str = "auto",
    run_name: str = "",
    concurrency: int = 64,
    structure_guard: bool | None = None,
    evaluation_frequency: str | None = None,
):
    import threading
    stop = threading.Event()
    output = Path("/runs") / run_name
    _live_engine["phase"] = "loading"

    def observe():
        import subprocess
        import traceback
        from recordings import save, now
        history = None
        previous = None
        while not stop.is_set():
            try:
                gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=5).strip().splitlines()[0].split(",")
                engine = _live_engine.get("engine")
                sample = dict(getattr(engine, "telemetry", {}))
                phase = _live_engine.get("phase", "loading")
                sample.setdefault("state", phase)
                if phase != "generating":
                    sample = {"state": phase}
                sample, previous = throughput_sample(sample, previous, time.monotonic())
                value = {"updatedAt": now(), "gpuUtilization": float(gpu[0]), "memoryUsedGiB": float(gpu[1])/1024, "memoryTotalGiB": float(gpu[2])/1024, **sample}
                with io_lock:
                    save(output / "inference-telemetry.json", value)
                    history_path = output / "inference-history.json"
                    if history is None:
                        history = json.loads(history_path.read_text()) if history_path.exists() else []
                    progress_path = output / "evaluation-status.json"
                    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                    history.append({**value, "step": progress.get("step"), "split": progress.get("split")})
                    history = history[-5000:]
                    save(history_path, history)
                    commit_results()
            except Exception:
                traceback.print_exc()
            stop.wait(10)
    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    try:
        return main_job(
            evaluate,
            attention_backend,
            run_name,
            concurrency,
            structure_guard,
            evaluation_frequency,
        )
    except Exception as exc:
        if (Path("/runs") / run_name / "status.json").exists():
            output = Path("/runs") / run_name
            plan = json.loads((output / "plan.json").read_text())
            message = "Student evaluation failed (" + type(exc).__name__ + "); see Modal logs."
            if plan.get("stream_evaluation", plan.get("evaluate_every_step", False)):
                from recordings import save, now

                save(
                    output / "evaluation-status.json",
                    {"state": "failed", "message": message, "updatedAt": now()},
                )
            else:
                from pipeline import publish

                publish(message, run_name=run_name, state="failed")
        raise
    finally:
        stop.set()
        observer.join(timeout=10)
        from recordings import save, now
        with io_lock:
            stopped = {"updatedAt": now(), "state": "stopped", "outputTokensPerSecond": 0.0}
            save(output / "inference-telemetry.json", stopped)
            history_path = output / "inference-history.json"
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            save(history_path, (history + [stopped])[-5000:])
        _live_engine.clear()
        commit_results()
        cache.commit()


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=28800,
    max_containers=1,
    retries=0,
    volumes={"/cache": cache, "/runs": results},
    secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
)
@track_gpu("inference")
def run(
    evaluate: bool = True,
    attention_backend: str = "auto",
    run_name: str = "",
    concurrency: int = 64,
    structure_guard: bool | None = None,
    evaluation_frequency: str | None = None,
):
    return execute_evaluation(
        evaluate, attention_backend, run_name, concurrency, structure_guard, evaluation_frequency
    )


@app.local_entrypoint()
def main(
    prepare: bool = False,
    run_name: str = "",
    attention_backend: str = "auto",
    concurrency: int = 64,
    structure_guard: bool = False,
    evaluation_frequency: str = "",
):
    if prepare:
        print(prepare_model.remote())
    else:
        if not run_name:
            raise ValueError("Provide --run-name explicitly")
        print(
            json.dumps(
                run.remote(
                    True,
                    attention_backend,
                    run_name,
                    concurrency,
                    True if structure_guard else None,
                    evaluation_frequency or None,
                ),
                indent=2,
            )
        )
