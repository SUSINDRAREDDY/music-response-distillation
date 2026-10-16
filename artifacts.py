"""Read-only access to published music assets; model checkpoints stay private."""

from pathlib import Path
from threading import Lock

import modal

app = modal.App("mimo-music-artifacts")
volume = modal.Volume.from_name("mimo-music-runs")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi==0.135.1", "modal==1.6.1")
    .add_local_file(Path(__file__).resolve().parent / "costs.py", "/root/costs.py")
)


def combined_state(training, evaluation):
    if "failed" in (training, evaluation):
        return "failed"
    if training in ("evaluating", "finished") and evaluation == "evaluating":
        return "evaluating"
    if training == "evaluating" and evaluation == "finished":
        return "finished"
    return training


def read_protocol_scores(directory, manifest):
    """Never combine student rewards from different generation protocols."""
    import json
    import math

    namespace = manifest.get("artifactNamespace", "native-fp8")
    if (
        not isinstance(namespace, str)
        or not namespace
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in namespace
        )
    ):
        raise ValueError("Invalid inference artifact namespace")
    # The worker publishes its complete score index atomically alongside each snapshot.
    # Reuse only an index with exactly the same protocol and namespace.
    index_path = directory / namespace / "evaluation.json"
    if index_path.exists():
        indexed = json.loads(index_path.read_text())
        if indexed.get("reference") == manifest.get("reference") and indexed.get("artifactNamespace") == namespace:
            return indexed.get("verifierScores", [])
    scores = [row for row in manifest.get("verifierScores", []) if row.get("version") == "teacher"]
    for score_file in (directory / namespace).glob("*/*/*/score.json"):
        split, prompt_id, version, _ = score_file.relative_to(directory / namespace).parts
        value = json.loads(score_file.read_text())
        reward = value.get("reward", 0)
        if (
            split in ("train", "validation")
            and isinstance(reward, (int, float))
            and math.isfinite(reward)
        ):
            scores = [
                row
                for row in scores
                if (row["promptId"], row["split"], row["version"]) != (prompt_id, split, version)
            ]
            scores.append(
                {
                    "promptId": prompt_id,
                    "split": split,
                    "version": version,
                    "reward": reward,
                    "accepted": value.get("accepted") is True,
                    "reason": value.get("reason") if isinstance(value.get("reason"), str) else None,
                }
            )
    return scores


@app.function(image=image, volumes={"/runs": volume}, cpu=1, memory=1024, scaledown_window=60)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    import mimetypes

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response

    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    lock = Lock()
    root = Path("/runs/public").resolve()
    cost_lock = Lock()
    cost_cache = {"at": 0, "value": None}
    evaluation_cache = {}

    @api.get("/live/costs.json")
    def costs():
        import json
        import time

        from costs import snapshot
        from fastapi.responses import JSONResponse

        with cost_lock:
            with lock:
                volume.reload()
            current_path = root / "live/current.json"
            run_id = (
                json.loads(current_path.read_text()).get("runId") if current_path.exists() else None
            )
            if (
                cost_cache["value"] is None
                or cost_cache["value"].get("runId") != run_id
                or time.monotonic() - cost_cache["at"] > 60
            ):
                cost_cache["value"] = snapshot()
                cost_cache["at"] = time.monotonic()
            return JSONResponse(cost_cache["value"], headers={"Cache-Control": "no-store"})

    def read(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.suffix not in (".json", ".abc", ".mid"):
            raise HTTPException(404)
        with lock:
            volume.reload()
            if not path.is_file():
                print(
                    f"Published asset absent: {path}; public root exists={root.exists()}",
                    flush=True,
                )
                raise HTTPException(404)
            content = path.read_bytes()
        return Response(
            content,
            media_type=mimetypes.guess_type(str(path))[0] or "application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    @api.get("/live/current.json")
    def live():
        import json

        from fastapi.responses import JSONResponse

        with lock:
            volume.reload()
            path = root / "live/current.json"
            if not path.exists():
                raise HTTPException(404)
            value = json.loads(path.read_text())
            name = value.get("runName", "")
            if name and all(c.isalnum() or c in "-_" for c in name):
                telemetry = Path("/runs") / name / "inference-telemetry.json"
                if telemetry.exists():
                    value["inferenceTelemetry"] = json.loads(telemetry.read_text())
                evaluation = Path("/runs") / name / "evaluation-status.json"
                if evaluation.exists():
                    progress = json.loads(evaluation.read_text())
                    epochs_path = Path("/runs") / name / "epoch-checkpoints.json"
                    if epochs_path.exists():
                        progress["readySteps"] = sorted(
                            int(path.parent.name.split("-")[-1])
                            for path in (Path("/runs") / name).glob("checkpoint-*/ready.json")
                        )
                    value["evaluationProgress"] = progress
                    value["trainingState"] = "finished" if (Path("/runs") / name / "result.json").exists() else value["state"]
                    value["inferenceState"] = progress["state"]
                    value["state"] = combined_state(value["state"], progress["state"])
                    value["activity"].append(
                        {"at": progress["updatedAt"], "message": progress["message"]}
                    )
            if name and all(c.isalnum() or c in "-_" for c in name):
                history_path = Path("/runs") / name / "inference-history.json"
                value["inferenceHistory"] = json.loads(history_path.read_text()) if history_path.exists() else []
            return JSONResponse(value, headers={"Cache-Control": "no-store"})

    @api.get("/runs/{run_id}/evaluation.json")
    def evaluation(run_id: str):
        if not run_id.isalnum():
            raise HTTPException(404)
        import json

        from fastapi.responses import JSONResponse

        import time
        with lock:
            cached = evaluation_cache.get(run_id)
            if cached and time.monotonic() - cached[0] < 15:
                return JSONResponse(cached[1], headers={"Cache-Control": "public, max-age=10"})
            volume.reload()
            directory = root / "runs" / run_id
            path = directory / "evaluation.json"
            if not path.is_file():
                raise HTTPException(404)
            manifest = json.loads(path.read_text())
            current_path = root / "live/current.json"
            if current_path.exists():
                current = json.loads(current_path.read_text())
                name = current.get("runName", "")
                if (
                    current.get("runId") == run_id
                    and name
                    and all(c.isalnum() or c in "-_" for c in name)
                ):
                    plan_path = Path("/runs") / name / "plan.json"
                    if plan_path.exists():
                        plan = json.loads(plan_path.read_text())
                        epochs = plan.get("epochs", 0)
                        total = current.get("totalSteps", 0)
                        if (
                            isinstance(epochs, int)
                            and epochs > 0
                            and isinstance(total, int)
                            and total > 0
                            and total % epochs == 0
                        ):
                            epochs_path = Path("/runs") / name / "epoch-checkpoints.json"
                            manifest["plannedCheckpoints"] = (
                                json.loads(epochs_path.read_text())
                                if epochs_path.exists()
                                and len(json.loads(epochs_path.read_text())) == epochs
                                else [
                                    {
                                        "id": f"step-{step}",
                                        "step": step,
                                        "epoch": (step - 1) // (total // epochs) + 1,
                                    }
                                    for step in range(total // epochs, total + 1, total // epochs)
                                ]
                            )
                            manifest["plannedTrainingCheckpoints"] = (
                                [
                                    {
                                        "id": f"step-{step}",
                                        "step": step,
                                        "epoch": (step - 1) // (total // epochs) + 1,
                                    }
                                    for step in range(1, total + 1)
                                ]
                                if manifest.get("reference", {}).get(
                                    "evaluationFrequency", plan.get("evaluation_frequency")
                                )
                                == "optimizer"
                                else manifest["plannedCheckpoints"]
                            )
                            if plan.get("training_reward_scope") == "optimizer_batch":
                                manifest["trainingRewardScope"] = "optimizer_batch"
                                for checkpoint in manifest["plannedTrainingCheckpoints"]:
                                    update_path = (
                                        Path("/runs")
                                        / name
                                        / f"checkpoint-{checkpoint['step']}"
                                        / "update-prompts.json"
                                    )
                                    if update_path.is_file():
                                        checkpoint["promptIds"] = json.loads(
                                            update_path.read_text()
                                        )["promptIds"]
            manifest["verifierScores"] = read_protocol_scores(directory, manifest)
            # Per-composition prompt ID lists duplicate the fixed protocol thousands of times.
            # Playback already loads full details from detailsUrl on demand.
            for key in ("recordings", "trainingRecordings"):
                for sample in manifest.get(key, []):
                    for composition in sample.get("compositions", []):
                        scoring = composition.get("scoring")
                        if scoring:
                            composition["scoring"] = {k: scoring[k] for k in
                                ("promptId", "promptSet", "split", "scorer", "protocol", "scope") if k in scoring}
            evaluation_cache[run_id] = (time.monotonic(), manifest)
        return JSONResponse(manifest, headers={"Cache-Control": "public, max-age=10"})

    @api.get("/runs/{run_id}/test-evaluation.json")
    def test_evaluation(run_id: str):
        if not run_id.isalnum():
            raise HTTPException(404)
        return read(f"runs/{run_id}/test-evaluation.json")

    @api.get("/music/runs/{path:path}")
    def music(path: str):
        if any(part in (".", "..") for part in Path(path).parts):
            raise HTTPException(404)
        # Publish only the playable score, MIDI and player metadata.
        if Path(path).name not in ("full.abc", "full.mid", "details.json"):
            raise HTTPException(404)
        return read(f"runs/{path}")

    return api
