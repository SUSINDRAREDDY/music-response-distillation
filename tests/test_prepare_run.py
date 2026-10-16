import json
from pathlib import Path

import prepare_run


def test_fresh_runs_have_unique_ids_balanced_disjoint_prompts_and_no_teacher_seeds(
    tmp_path, monkeypatch
):
    source = Path(prepare_run.ROOT)
    (tmp_path / "config.json").write_text((source / "config.json").read_text())
    (tmp_path / "data").symlink_to(source / "data", target_is_directory=True)
    monkeypatch.setattr(prepare_run, "ROOT", tmp_path)
    first = prepare_run.prepare(seed=42)
    second = prepare_run.prepare(seed=42)
    assert first["run_id"] != second["run_id"]
    assert first["run_name"] != second["run_name"]
    assert first["train_tasks"] == second["train_tasks"]
    assert len(first["train_tasks"]) == 50
    assert len(first["validation_tasks"]) == 10
    assert sum(t["length"] == "short" for t in first["train_tasks"]) == 25
    assert sum(t["length"] == "short" for t in first["validation_tasks"]) == 5
    assert not {t["task_id"] for t in first["train_tasks"]} & {
        t["task_id"] for t in first["validation_tasks"]
    }
    assert first["teacher_concurrency"] == 10
    assert first["teacher_validation"] is False
    directory = tmp_path / "runs" / first["run_name"]
    assert [p.name for p in directory.iterdir()] == ["plan.json"]
    assert json.loads((directory / "plan.json").read_text()) == first


def test_new_plan_uses_epoch_comparisons_and_baseline(tmp_path, monkeypatch):
    source = Path(prepare_run.ROOT)
    (tmp_path / "config.json").write_text((source / "config.json").read_text())
    (tmp_path / "data").symlink_to(source / "data", target_is_directory=True)
    monkeypatch.setattr(prepare_run, "ROOT", tmp_path)
    plan = prepare_run.prepare(seed=42)
    assert plan["evaluation_frequency"] == "epoch"
    assert plan["training_reward_scope"] == "fixed_training_set"
    assert plan["baseline_generation"] is True


def test_all_training_prompts_preserve_unequal_lengths(tmp_path, monkeypatch):
    source = Path(prepare_run.ROOT)
    (tmp_path / "config.json").write_text((source / "config.json").read_text())
    (tmp_path / "data").symlink_to(source / "data", target_is_directory=True)
    monkeypatch.setattr(prepare_run, "ROOT", tmp_path)
    plan = prepare_run.prepare(seed=42, all_train=True)
    assert len(plan["train_tasks"]) == 806
    assert sum(t["length"] == "short" for t in plan["train_tasks"]) == 408
    assert sum(t["length"] == "long" for t in plan["train_tasks"]) == 398
    assert len({t["task_id"] for t in plan["train_tasks"]}) == 806
    assert not {t["task_id"] for t in plan["train_tasks"]} & {
        t["task_id"] for t in plan["validation_tasks"]
    }
