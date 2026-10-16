import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "639865fd3374018d6cb29b9fb82dd531406fcf5f"
SHA256 = "8fa2c95fb008df032eae184f3304dd3a73e3d25727e3aac0a715484859fa4856"
URL = (
    f"https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/resolve/{REVISION}/music.parquet"
)


def split_for(prompt):
    text = " ".join(prompt.split()).casefold()
    bucket = int(hashlib.sha256(text.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def prepare():
    import pyarrow.parquet as pq

    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    payload = urllib.request.urlopen(URL, timeout=60).read()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("Dataset checksum mismatch")
    source = directory / "source.parquet"
    source.write_bytes(payload)
    rows = pq.read_table(source).to_pylist()
    splits = {name: [] for name in ("train", "validation", "test")}
    ids = set()
    for row in rows:
        task_id = row["extra_info"]["src_id"]
        if task_id in ids:
            raise ValueError(f"Duplicate task: {task_id}")
        ids.add(task_id)
        text = "\n".join(message["content"] for message in row["prompt"])
        splits[split_for(text)].append(
            {
                "prompt": row["prompt"],
                "task_id": task_id,
                **row["extra_info"],
            }
        )
    for name, tasks in splits.items():
        (directory / f"{name}.jsonl").write_text(
            "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks)
        )
        for length in ("short", "long"):
            (directory / f"{name}-{length}.jsonl").write_text(
                "".join(
                    json.dumps(task, ensure_ascii=False) + "\n"
                    for task in tasks
                    if task["length"] == length
                )
            )
    manifest = {
        "url": URL,
        "revision": REVISION,
        "sha256": SHA256,
        "counts": {name: len(tasks) for name, tasks in splits.items()},
        "counts_by_length": {
            name: {
                length: sum(task["length"] == length for task in tasks)
                for length in ("short", "long")
            }
            for name, tasks in splits.items()
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def load_tasks(directory, split, limit=None, length=None):
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split: {split}")
    if length not in (None, "short", "long"):
        raise ValueError(f"Unknown length: {length}")
    rows = [
        json.loads(line) for line in (Path(directory) / f"{split}.jsonl").read_text().splitlines()
    ]
    if length is not None:
        rows = [row for row in rows if row["length"] == length]
    if not rows:
        raise ValueError(f"No tasks in {split} for length={length}")
    return rows[:limit] if limit else rows


def training_data_path(directory, split="train"):
    filtered = Path(directory) / f"{split}-sft.jsonl"
    return filtered if filtered.exists() else Path(directory) / f"{split}.jsonl"


def filter_training_rows(rows, tokenizer, max_length, template):
    accepted, excluded = [], []
    for row in rows:
        length = len(tokenizer.apply_chat_template(
            row["prompt"] + row["completion"], **template, return_dict=True,
        )["input_ids"])
        if length > max_length:
            excluded.append({"task_id": row["task_id"], "tokens": length,
                             "reason": f"Exceeds student context limit ({max_length} tokens)"})
        else:
            accepted.append(row)
    return accepted, excluded


if __name__ == "__main__":
    prepare()
