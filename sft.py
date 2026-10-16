"""Stock TRL completion-only SFT with verified examples and PEFT LoRA."""

import json
import math
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
cache = modal.Volume.from_name("mimo-music-model-cache", create_if_missing=True)
results = modal.Volume.from_name("mimo-music-runs", create_if_missing=True)
image = modal.Image.debian_slim()
if modal.is_local():
    image = (
        modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
        .pip_install_from_requirements(ROOT / "requirements-sft.txt")
        .apt_install("build-essential")
        .pip_install("ninja==1.13.0", "packaging", "setuptools", "wheel")
        .run_commands(
            "CC=gcc CXX=g++ MAX_JOBS=2 CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install causal-conv1d==1.7.0 --no-build-isolation",
            "python -c 'from causal_conv1d import causal_conv1d_fn, causal_conv1d_update'",
        )
        .add_local_file(ROOT / "setup_scorer.py", "/root/setup_scorer.py", copy=True)
        .run_commands("python /root/setup_scorer.py")
        .env(
            {
                "ABC2MIDI_BIN": "/root/tools/abc2midi",
                "HF_HOME": "/cache/huggingface",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "4",
                "TRITON_CACHE_DIR": "/cache/triton-2.13",
                "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor-2.13",
            }
        )
        .add_local_file(ROOT / "sft.py", "/root/sft.py")
        .add_local_file(ROOT / "config.json", "/root/config.json")
        .add_local_file(ROOT / "recordings.py", "/root/recordings.py")
        .add_local_file(ROOT / "teacher.py", "/root/teacher.py")
        .add_local_file(ROOT / "codex_accounts.py", "/root/codex_accounts.py")
        .add_local_file(ROOT / "data.py", "/root/data.py")
        .add_local_file(ROOT / "reward.py", "/root/reward.py")
        .add_local_dir(ROOT / "vendor", "/root/vendor", ignore=["**/__pycache__/**"])
    )


def training_prompt_ids(inputs, token_index):
    """Identify consumed examples from their unpadded tokens, after shuffling."""
    if "attention_mask" not in inputs:
        raise ValueError("Update tracking requires un-packed, padded batches")
    return [
        token_index[tuple(token for token, valid in zip(row, mask, strict=True) if valid)]
        for row, mask in zip(
            inputs["input_ids"].tolist(), inputs["attention_mask"].tolist(), strict=True
        )
    ]


def build_trainer(
    data_dir,
    output_dir,
    run_name,
    examples,
    every_step=False,
    teacher_validation=False,
    track_update_prompts=False,
):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    from data import training_data_path

    model_config = json.loads((ROOT / "config.json").read_text())
    settings = model_config["sft"]["trainer"].copy()
    train_rows = [
        json.loads(line) for line in training_data_path(data_dir).read_text().splitlines()
    ]
    eval_rows = (
        [
            json.loads(line)
            for line in training_data_path(data_dir, "validation").read_text().splitlines()
        ]
        if teacher_validation
        else []
    )
    if not train_rows or (teacher_validation and not eval_rows):
        raise ValueError("Require verified train and validation teacher answers")
    if len(train_rows) != examples or {r["task_id"] for r in train_rows} & {
        r["task_id"] for r in eval_rows
    }:
        raise ValueError("Training count differs from the plan or overlaps validation")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model"], revision=model_config["revision"]
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    lengths = []
    template = model_config.get("chat_template_kwargs", {})
    for row in train_rows + eval_rows:
        row["chat_template_kwargs"] = template
        lengths.append(
            len(
                tokenizer.apply_chat_template(
                    row["prompt"] + row["completion"],
                    **template,
                    return_dict=True,
                )["input_ids"]
            )
        )
    needed = math.ceil(max(lengths) / 256) * 256
    if needed > settings["max_length"]:
        raise ValueError(f"Longest teacher answer needs {needed} tokens; increase max_length")
    settings.update(max_length=needed, eval_strategy="epoch" if teacher_validation else "no")
    if every_step:
        settings.update(save_strategy="steps", save_steps=1, save_total_limit=None)
    args = SFTConfig(output_dir=output_dir, run_name=run_name, **settings)
    set_seed(args.seed)
    quantization = BitsAndBytesConfig(
        **model_config["quantization"], bnb_4bit_compute_dtype=torch.bfloat16
    )
    model, loading = AutoModelForCausalLM.from_pretrained(
        model_config["model"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if loading["missing_keys"] or loading.get("mismatched_keys"):
        raise RuntimeError(f"Student checkpoint did not load completely: {loading}")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
    )

    class UpdateTrackedTrainer(SFTTrainer):
        def training_step(self, model, inputs, *args, **kwargs):
            ids = training_prompt_ids(inputs, self.token_prompt_index)
            loss = super().training_step(model, inputs, *args, **kwargs)
            self.update_prompt_ids.extend(ids)
            return loss

    trainer = (UpdateTrackedTrainer if track_update_prompts else SFTTrainer)(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows) if teacher_validation else None,
        peft_config=LoraConfig(**model_config["sft"]["lora"]),
    )
    actual = [len(row["input_ids"]) for row in trainer.train_dataset]
    if len(actual) != len(train_rows) or max(actual) != max(lengths[: len(train_rows)]):
        raise RuntimeError("TRL tokenization differs from preflight; inspect before training")
    if track_update_prompts:
        if not every_step or args.packing or args.padding_free or args.world_size != 1:
            raise ValueError(
                "Update reward tracking requires single-GPU un-packed per-step checkpoints"
            )
        trainer.token_prompt_index = {
            tuple(row["input_ids"]): task["task_id"]
            for row, task in zip(trainer.train_dataset, train_rows, strict=True)
        }
        if len(trainer.token_prompt_index) != len(train_rows):
            raise ValueError("Cannot uniquely identify training examples by tokens")
        trainer.update_prompt_ids = []
    example = trainer.train_dataset[0]
    prefix = tokenizer.apply_chat_template(
        train_rows[0]["prompt"],
        add_generation_prompt=True,
        **template,
        return_dict=True,
    )["input_ids"]
    labels = trainer.data_collator([example])["labels"][0].tolist()
    if not all(label == -100 for label in labels[: len(prefix)]) or not any(
        label != -100 for label in labels[len(prefix) : len(example["input_ids"])]
    ):
        raise RuntimeError("TRL completion-only loss mask failed preflight")
    return trainer, {
        "settings": args.to_dict(),
        "wandb_project": model_config["wandb_project"],
        "model": model_config["model"],
        "revision": model_config["revision"],
        "quantization": quantization.to_dict(),
        "chat_template_kwargs": template,
        "teacher": {"model": "gpt-6.1-sol", "reasoning_effort": "low"},
        "lora": model_config["sft"]["lora"],
        "purpose": "sft",
        "train_task_ids": [row["task_id"] for row in train_rows],
        "validation_task_ids": [row["task_id"] for row in eval_rows],
        "train_examples": len(train_rows),
        "validation_examples": len(eval_rows),
        "max_tokens": max(lengths),
        "max_train_tokens": max(lengths[: len(train_rows)]),
        "mean_train_tokens": sum(lengths[: len(train_rows)]) / len(train_rows),
    }
