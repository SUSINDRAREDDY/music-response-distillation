# Music response distillation

Teacher-generated ABC music → verification → completion-only QLoRA SFT → held-out evaluation.
This is **off-policy response distillation**, using hard teacher-answer tokens, not teacher logits or reinforcement learning.

The student is `Qwen/Qwen3.8-27B` at the revision in `config.json`. The teacher is
`gpt-6.1-sol` with low reasoning effort. Of 806 training prompts, 607 teacher answers
passed the implemented checks and were used for SFT; 199 were rejected.

## Results

Saved run: `music-20261009-150609-493551` (`49355152800c`).

| Held-out validation | Base | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|---:|
| Mean verifier reward | 0.086 | 0.160 | 0.191 | 0.284 |
| Passes all implemented checks / 105 | 5 | 10 | 9 | 16 |
| Zero reward / 105 | 83 | 78 | 74 | 61 |

Per-prompt scores and exact means are in `evaluation/`. These are measured outputs
from one run, not confidence intervals or evidence of human-rated musical quality.
The model still struggles with composition length, beats per bar, and coordinating
voices. Positive music reward does not imply exact instruction compliance. The
structural generation guard counts completed lines and can miss a runaway unfinished
line. Aborted generations remain zero-reward outcomes in the denominator.

Training used 3 epochs, batch size 1, accumulation 16, learning rate 1e-4,
NF4 QLoRA (rank 16, alpha 32), completion-only loss, and 114 optimizer updates.
The configured context ceiling is 16,384; the saved run dynamically selected 11,008
from the longest 11,000-token training sequence. Evaluation used greedy generation,
8,192 maximum new tokens, seed 42, and the recorded grouped/guarded artifact namespace.

## Data

Dataset: [suunny/music-teacher-distillation](https://huggingface.co/datasets/suunny/music-teacher-distillation).
The default configuration contains the exact 607 training examples. Rejected teacher
answers are a separate configuration with verifier scores and reasons. Validation
(105) and test (89) contain only prompts and belong to a separate held-out configuration.
Do not combine these held-out prompts with the training data.

Prompts originate from [XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss),
revision `639865fd3374018d6cb29b9fb82dd531406fcf5f`. The pinned scorer is vendored with
its Apache-2.0 license and hash manifest. See `NOTICE`.

## Local setup and verification

Use Python 3.12+ and a C compiler for abc2midi:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-local.txt
python setup_scorer.py
python data.py
ABC2MIDI_BIN="$PWD/tools/abc2midi" python -m pytest tests test_costs.py test_epoch_results.py -q
```

`setup_scorer.py` builds pinned upstream abc2midi locally; its binary is not bundled.
GPU dependencies are pinned separately in `requirements-sft.txt`. Local tests do not
establish GPU runtime compatibility or music quality.

## Pipeline

`data.py` downloads and splits the source prompts. `teacher.py` verifies teacher answers.
`sft.py` builds the TRL trainer: prompt/padding tokens are excluded from loss;
backpropagation updates LoRA weights. `pipeline.py` coordinates collection and training.
`inference.py` and `inference_engine.py` evaluate checkpoints. `recordings.py` and
`artifacts.py` provide playback artifacts. `codex_*` contain the historical authenticated
collector implementation; no credentials, sessions, account logs, or private volumes
are included. Collection requires an appropriately authorized provider account and
compliance with its applicable terms; the repository does not grant provider rights.

To prepare a new plan (local only):

```sh
python data.py
python prepare_run.py --all-train
```

New plans default to a 10-prompt validation pilot; the reported run used all 105
validation prompts. Reproducing its evaluation requires that full set. Follow the
printed run environment variables. Modal execution requires your own authenticated
workspace, `wandb` secret, and teacher auth volume. `modal deploy inference.py`,
`modal deploy artifacts.py`, and `modal run --detach pipeline.py` execute cloud work
and incur charges. No automatic launch occurs during setup or tests.

`scripts/export_dataset.py` reconstructs the public dataset from downloaded original
run files, verifying manifest IDs and split separation while exporting only an explicit
allowlist. The private source directory is deliberately excluded from this repository.
