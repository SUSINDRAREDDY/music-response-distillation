# Running the pipeline

## Local setup and verification

Use Python 3.12+ and a C compiler for abc2midi:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-local.txt
python setup_scorer.py
python data.py
ABC2MIDI_BIN="$PWD/tools/abc2midi" python -m pytest -q
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
python prepare_run.py --all-train --eval-samples 105
```

Without `--eval-samples 105`, new plans default to a 10-prompt validation pilot.
The command above selects the full validation set used for the reported results. Follow the
printed run environment variables. Modal execution requires your own authenticated
workspace, `wandb` secret, and teacher auth volume. `modal deploy inference.py`,
`modal deploy artifacts.py`, and `modal run --detach pipeline.py` execute cloud work
and incur charges. No automatic launch occurs during setup or tests.

`scripts/export_dataset.py` reconstructs the public dataset from downloaded original
run files, verifying manifest IDs and split separation while exporting only an explicit
allowlist. The private source directory is deliberately excluded from this repository.
