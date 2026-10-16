"""Prepare a fresh balanced training/held-out run locally. Does not launch Modal jobs."""

import argparse
import json
import random
import secrets
from datetime import UTC, datetime

from data import ROOT, load_tasks


def prepare(seed=None, samples=50, all_train=False, eval_samples=10):
    if not all_train and (samples < 2 or samples % 2):
        raise ValueError("Training samples must be a positive even count")
    if eval_samples < 1:
        raise ValueError("Evaluation samples must be positive")
    seed = secrets.randbits(32) if seed is None else seed
    rng = random.Random(seed)
    run_id = secrets.token_hex(6)
    name = f"music-{datetime.now(UTC):%Y%m%d-%H%M%S}-{run_id[:6]}"
    config = json.loads((ROOT / "config.json").read_text())
    plan = {
        "run_name": name,
        "run_id": run_id,
        "selection_seed": seed,
        "epochs": config["sft"]["trainer"]["num_train_epochs"],
        "gradient_accumulation": config["sft"]["trainer"]["gradient_accumulation_steps"],
        "generation": {"do_sample": False, "max_new_tokens": 8192, "seed": 42},
        "scorer_revision": "a2ad9f6160b03ff2d47e59832bfb6b289f37c917",
        "teacher_attempts": 1,
        "teacher_concurrency": 10,
        "teacher_validation": False,
        "validation_loss_ids": [],
        "stream_evaluation": True,
        "evaluation_frequency": "epoch",
        "training_reward_scope": "fixed_training_set",
        "baseline_generation": True,
        "structure_guard": True,
    }
    for split, count in (("train", samples // 2), ("validation", eval_samples // 2)):
        if split == "train" and all_train:
            tasks = load_tasks(ROOT / "data", split)
            rng.shuffle(tasks)
            plan["train_tasks"] = tasks
            continue
        plan[split + "_tasks"] = [
            task
            for length in ("short", "long")
            for task in rng.sample(
                load_tasks(ROOT / "data", split, length=length),
                count + (1 if split == "validation" and length == "short" and eval_samples % 2 else 0),
            )
        ]
    directory = ROOT / "runs" / name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--samples", type=int, default=50)
    selection.add_argument("--all-train", action="store_true")
    parser.add_argument("--eval-samples", type=int, default=10)
    args = parser.parse_args()
    plan = prepare(samples=args.samples, all_train=args.all_train, eval_samples=args.eval_samples)
    print(
        f"Prepared {plan['run_name']}: {len(plan['train_tasks'])} training prompts, "
        f"{len(plan['validation_tasks'])} held-out prompts."
    )
    print(f"export MUSIC_RUN_NAME={plan['run_name']}")
    print(f"export MUSIC_RUN_ID={plan['run_id']}")
    print("modal run --detach pipeline.py")
