"""Publish aggregate music costs only, never prompts, credentials or other apps."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Standard API-equivalent pricing, verified 2026-10-07. ChatGPT login is not API billing.
PRICING = {"input": 2.0, "cached": 0.1, "write": 2.5, "output": 10.0}
PRICING_URL = "https://developers.openai.com/api/docs/models/gpt-6.1-sol"


def teacher_usage(root: Path, start: datetime, end: datetime, run_name=None):
    totals = dict(input=0, cached=0, write=0, output=0)
    seen = set()
    calls = missing = 0
    cost = 0.0
    paths = (
        (root / run_name).glob("teacher/*/*/*/events.jsonl")
        if run_name
        else root.glob("music*/teacher/*/*/*/events.jsonl")
    )
    for path in paths:
        if not run_name and not start.timestamp() <= path.stat().st_mtime < end.timestamp():
            continue
        thread = str(path.relative_to(root))
        turns = 0
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "thread.started":
                thread = event.get("thread_id", thread)
            if event.get("type") != "turn.completed":
                continue
            turns += 1
            key = (thread, turns)
            if key in seen:
                continue
            usage = event.get("usage", {})
            if (
                not isinstance(usage, dict)
                or "input_tokens" not in usage
                or "output_tokens" not in usage
            ):
                missing += 1
                continue
            fields = {
                name: usage.get(field, 0)
                for name, field in (
                    ("input", "input_tokens"),
                    ("cached", "cached_input_tokens"),
                    ("write", "cache_write_input_tokens"),
                    ("output", "output_tokens"),
                )
            }
            if not all(isinstance(v, int) and v >= 0 for v in fields.values()):
                missing += 1
                continue
            seen.add(key)
            calls += 1
            for name, value in fields.items():
                totals[name] += value
            long_context = fields["input"] > 272_000
            input_multiplier = 2 if long_context else 1
            output_multiplier = 1.5 if long_context else 1
            cost += (
                (
                    max(0, fields["input"] - fields["cached"] - fields["write"]) * PRICING["input"]
                    + fields["cached"] * PRICING["cached"]
                    + fields["write"] * PRICING["write"]
                )
                * input_multiplier
                + fields["output"] * PRICING["output"] * output_multiplier
            ) / 1_000_000
        if not turns:
            missing += 1
    return {
        **totals,
        "total": totals["input"] + totals["output"],
        "calls": calls,
        "missingCalls": missing,
        "estimatedCost": cost,
        "model": "gpt-6.1-sol",
        "pricingUrl": PRICING_URL,
        "ratesPerMillion": PRICING,
        "pricingVerifiedAt": "2026-10-07",
    }


def modal_usage(rows, rates):
    resources = {}
    stages = {}
    for row in rows:
        if row.environment_name != "main" or row.description not in {
            "mimo-music-pipeline",
            "mimo-music-inference",
            "mimo-music-artifacts",
        }:
            continue
        stage = (
            "Inference"
            if row.description == "mimo-music-inference"
            else "Training and collection"
            if row.description == "mimo-music-pipeline"
            else "Other music services"
        )
        stages[stage] = stages.get(stage, Decimal(0)) + sum(
            row.cost_by_resource.values(), Decimal(0)
        )
        for resource, cost in row.cost_by_resource.items():
            resources[resource] = resources.get(resource, Decimal(0)) + cost
    result = []
    for resource, cost in sorted(resources.items()):
        key = {
            "CPU": "cpu_hour_cost",
            "Memory": "mem_gib_hour_cost",
            "CPU (Sandbox)": "cpu_hour_cost_sandbox",
            "Memory (Sandbox)": "mem_gib_hour_cost_sandbox",
        }.get(resource)
        if key is None:
            candidate = "gpu_hour_cost_" + resource.lower().replace(" ", "_")
            key = candidate if candidate in rates else None
        rate = rates.get(key) if key else None
        hours = float(cost / rate) if rate and rate > 0 else None
        result.append(
            {
                "name": resource,
                "cost": float(cost),
                "hours": hours,
                "unit": "core-h"
                if resource.startswith("CPU")
                else "GiB-h"
                if resource.startswith("Memory")
                else "GPU-h"
                if key
                else None,
            }
        )
    return {
        "reportedCost": float(sum(resources.values(), Decimal(0))),
        "resources": result,
        "stages": [{"name": name, "cost": float(cost)} for name, cost in sorted(stages.items())],
    }


def reset_run_costs(output, run_id):
    """Persist a billing baseline once. Resuming a run never resets its costs."""
    import modal
    from recordings import save

    path = output / "cost-baseline.json"
    if path.exists():
        return
    now = datetime.now(UTC)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    baseline = {
        "runId": run_id,
        "startedAt": now.isoformat(),
        "billingStart": start.isoformat(),
        "modal": None,
    }
    try:
        workspace = modal.Workspace.from_context()
        rows = workspace.billing.report(start=start, end=start + timedelta(days=1), resolution="d")
        baseline["modal"] = modal_usage(rows, workspace.billing.rates())
    except Exception as error:
        print(f"Music billing baseline unavailable: {type(error).__name__}", flush=True)
    save(path, baseline)


def subtract_baseline(current, baseline):
    result = {**current}
    for group in ("resources", "stages"):
        previous = {row["name"]: row["cost"] for row in baseline[group]}
        result[group] = []
        for row in current[group]:
            cost = max(0, row["cost"] - previous.get(row["name"], 0))
            value = {**row, "cost": cost}
            if "hours" in row:
                value["hours"] = (
                    row["hours"] * cost / row["cost"]
                    if row["cost"] and row["hours"] is not None
                    else None
                )
            result[group].append(value)
    result["reportedCost"] = sum(row["cost"] for row in result["resources"])
    return result


def gpu_runtime(output, now, hourly_rate=None):
    """Estimate GPU-only spend from observed worker lifetimes, never billing deltas."""
    rows = []
    for stage in ("training", "inference"):
        intervals = [json.loads(path.read_text()) for path in
                     (output / "gpu-runtime").glob(stage + "-*.json")]
        if not intervals:
            continue
        seconds = sum(max(0, row["elapsedSeconds"]) for row in intervals)
        unfinished = [row for row in intervals if not row.get("endedAt")]
        active = any((now - datetime.fromisoformat(row["updatedAt"])).total_seconds() < 120
                     for row in unfinished)
        rows.append({
            "stage": stage, "gpu": "H100",
            "startedAt": min(row["startedAt"] for row in intervals),
            "endedAt": None if unfinished else max(row["endedAt"] for row in intervals),
            "state": "running" if active else "unconfirmed" if unfinished else "finished",
            "seconds": seconds, "hourlyRate": hourly_rate,
            "estimatedCost": seconds / 3600 * hourly_rate if hourly_rate is not None else None,
        })
    return rows


def snapshot(root=Path("/runs")):
    import modal

    now = datetime.now(UTC)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    current_path = root / "public/live/current.json"
    current = json.loads(current_path.read_text()) if current_path.exists() else {}
    run_name = current.get("runName")
    if run_name and (Path(run_name).name != run_name or run_name in {".", ".."}):
        raise ValueError("Invalid cost run name")
    baseline_path = root / run_name / "cost-baseline.json" if run_name else None
    baseline = (
        json.loads(baseline_path.read_text()) if baseline_path and baseline_path.exists() else None
    )
    if baseline and baseline["runId"] != current.get("runId"):
        raise ValueError("Cost baseline belongs to a different run")
    result = {
        "updatedAt": now.isoformat(),
        "day": start.date().isoformat(),
        "currency": "USD",
        "scope": "run" if run_name else "day",
        "runId": current.get("runId"),
        "startedAt": baseline["startedAt"] if baseline else current.get("createdAt"),
        "model": teacher_usage(root, start, end, run_name),
        "modal": None,
    }
    result["gpuWorkers"] = gpu_runtime(root / run_name, now) if run_name else []
    try:
        workspace = modal.Workspace.from_context()
        rates = workspace.billing.rates()
        rate = rates.get("gpu_hour_cost_h100")
        if run_name and rate is not None:
            result["gpuWorkers"] = gpu_runtime(root / run_name, now, float(rate))
    except Exception as error:
        print(f"GPU rate unavailable: {type(error).__name__}", flush=True)
    # Never substitute earlier daily spend for a missing run baseline.
    if run_name and (not baseline or baseline["modal"] is None):
        return result
    try:
        workspace = modal.Workspace.from_context()
        if baseline:
            start = datetime.fromisoformat(baseline["billingStart"])
        rows = workspace.billing.report(start=start, end=end, resolution="d")
        result["modal"] = modal_usage(rows, workspace.billing.rates())
        if baseline:
            result["modal"] = subtract_baseline(result["modal"], baseline["modal"])
    except Exception as error:
        # No account details or error strings cross the public API.
        print(f"Music billing unavailable: {type(error).__name__}", flush=True)
    return result
