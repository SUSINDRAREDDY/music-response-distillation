import ast
import unittest
import threading
from pathlib import Path

# Load only the pure scheduling helpers; importing inference configures Modal images.
module = ast.parse((Path(__file__).resolve().parents[1] / "inference.py").read_text())
helpers = ast.Module(
    body=[
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in (
            "evaluation_schedule",
            "pending_tasks",
            "evaluation_log_directory",
            "checkpoint_mean",
            "wait_for_file",
            "reload_results",
            "zero_rate_warmup_checkpoint",
            "check_adapter",
            "optimizer_batch_tasks",
        )
    ],
    type_ignores=[],
)
namespace = {"io_lock": threading.RLock()}
exec(compile(helpers, "inference.py", "exec"), namespace)


class EpochResultsTest(unittest.TestCase):
    def test_all_three_epochs_generate_both_datasets_in_order(self):
        train = [{"task_id": f"t{i}"} for i in range(20)]
        val = [{"task_id": f"v{i}"} for i in range(10)]
        jobs = namespace["evaluation_schedule"]([5, 10, 15], train, val)
        self.assertEqual(
            [(step, split) for step, split, _ in jobs],
            [
                (0, "validation"),
                (5, "train"),
                (5, "validation"),
                (10, "train"),
                (10, "validation"),
                (15, "train"),
                (15, "validation"),
            ],
        )
        for step in (5, 10, 15):
            self.assertEqual(
                next(tasks for s, split, tasks in jobs if s == step and split == "train"), train
            )
        self.assertEqual(sum(len(tasks) for _, _, tasks in jobs), 100)

    def test_streaming_scores_only_epoch_checkpoints(self):
        train = [{"task_id": "t"}]
        val = [{"task_id": "v"}]
        jobs = namespace["evaluation_schedule"](range(1, 16), train, val, True, [5, 10, 15])
        self.assertEqual([step for step, split, _ in jobs if split == "validation"], [5, 10, 15])
        self.assertEqual([step for step, split, _ in jobs if split == "train"], [5, 10, 15])
        for step in (5, 10, 15):
            i = next(i for i, (s, split, _) in enumerate(jobs) if s == step and split == "train")
            self.assertEqual(jobs[i + 1], (step, "validation", val))

    def test_partial_resume_keeps_failed_zero_reward_outputs_and_split_identity(self):
        tasks = [{"task_id": "a"}, {"task_id": "b"}, {"task_id": "c"}]
        manifest = {
            "verifierScores": [
                {
                    "promptId": "a",
                    "split": "train",
                    "version": "step-5",
                    "reward": 0,
                    "accepted": False,
                },
                {"promptId": "b", "split": "validation", "version": "step-5", "reward": 0.5},
                {"promptId": "c", "split": "train", "version": "step-10", "reward": 0.7},
            ]
        }
        self.assertEqual(namespace["pending_tasks"](manifest, tasks, "train", "step-5"), tasks[1:])
        self.assertEqual(namespace["pending_tasks"](manifest, tasks, "train", "step-15"), tasks)

    def test_reward_denominator_includes_invalid_zero_and_requires_all_ids(self):
        rows = [
            {
                "promptId": "a",
                "split": "train",
                "version": "step-5",
                "reward": 0,
                "accepted": False,
            },
            {
                "promptId": "b",
                "split": "train",
                "version": "step-5",
                "reward": 0.8,
                "accepted": True,
            },
        ]
        tasks = [{"task_id": "a"}, {"task_id": "b"}]
        self.assertEqual(namespace["checkpoint_mean"](rows, tasks, "train", "step-5"), 0.4)
        with self.assertRaises(RuntimeError):
            namespace["checkpoint_mean"](rows[1:], tasks, "train", "step-5")

    def test_wandb_directory_is_outside_reload_volume_and_readiness_reload_survives(self):
        import json, time, tempfile
        from unittest.mock import Mock
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            marker = output / "ready.json"
            (output / "status.json").write_text(json.dumps({"state": "failed"}))
            (output / "result.json").write_text("{}")
            logs = Path(namespace["evaluation_log_directory"]("test"))
            self.assertFalse(logs.is_relative_to("/runs"))
            volume = Mock()
            volume.reload.side_effect = lambda: marker.write_text("{}")
            namespace.update(results=volume, json=json, time=time)
            with (logs / "debug.log").open("w"):
                namespace["wait_for_file"](marker, output, timeout=1)
            volume.reload.assert_called_once()

    def test_completed_training_reuses_saved_accepted_example_count_and_epoch_weights(self):
        import json, math, tempfile
        from pathlib import Path

        source = ast.parse((Path(__file__).resolve().parents[1] / "pipeline.py").read_text())
        fn = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "completed_training"
        )
        env = {"json": json, "math": math, "ROOT": Path(__file__).resolve().parents[1]}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "pipeline.py", "exec"), env)
        config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "plan.json").write_text(json.dumps({"run_id": "saved", "epochs": 3}))
            (out / "settings.json").write_text(
                json.dumps(
                    {
                        "model": config["model"],
                        "revision": config["revision"],
                        "train_examples": 17,
                        "settings": {
                            "gradient_accumulation_steps": 4,
                            "per_device_train_batch_size": 1,
                        },
                    }
                )
            )
            result = {"optimizer_steps": 15, "train": {"epoch": 3}}
            (out / "result.json").write_text(json.dumps(result))
            for step in (5, 10, 15):
                checkpoint = out / f"checkpoint-{step}"
                checkpoint.mkdir()
                for name in ("adapter_model.safetensors", "adapter_config.json"):
                    (checkpoint / name).write_text("{}")
            self.assertEqual(env["completed_training"](out, "saved", 4), result)
            (out / "checkpoint-10/adapter_model.safetensors").unlink()
            with self.assertRaises(ValueError):
                env["completed_training"](out, "saved", 4)

    def test_ready_marker_follows_complete_epoch_save_and_commit(self):
        import json, tempfile
        from types import SimpleNamespace
        from pathlib import Path

        source = ast.parse((Path(__file__).resolve().parents[1] / "pipeline.py").read_text())
        trainer = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_student"
        )
        cls = next(
            node
            for node in trainer.body
            if isinstance(node, ast.ClassDef) and node.name == "Progress"
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            checkpoint = output / "checkpoint-5"
            checkpoint.mkdir()
            events = []

            def save(path, value):
                events.append(path.name)
                path.write_text(json.dumps(value))

            volume = SimpleNamespace(commit=lambda: events.append("commit"))
            env = {
                "TrainerCallback": object,
                "output": output,
                "json": json,
                "save": save,
                "results": volume,
                "plan": {},
            }
            exec(compile(ast.Module(body=[cls], type_ignores=[]), "pipeline.py", "exec"), env)
            callback = env["Progress"]()
            state = SimpleNamespace(global_step=5, epoch=1.0)
            with self.assertRaises(RuntimeError):
                callback.on_save(None, state, None)
            self.assertFalse((checkpoint / "ready.json").exists())
            for name in (
                "adapter_model.safetensors",
                "adapter_config.json",
                "trainer_state.json",
                "optimizer.pt",
                "scheduler.pt",
            ):
                (checkpoint / name).write_text("{}")
            callback.on_save(None, state, None)
            self.assertEqual(events, ["commit", "epoch-checkpoints.json", "ready.json", "commit"])
            self.assertEqual(
                json.loads((output / "epoch-checkpoints.json").read_text()),
                [{"id": "step-5", "step": 5, "epoch": 1}],
            )
            env["plan"] = {"training_reward_scope": "optimizer_batch"}
            env["trainer"] = SimpleNamespace(update_prompt_ids=["a"])
            callback.on_save(
                SimpleNamespace(per_device_train_batch_size=1, gradient_accumulation_steps=4),
                state,
                None,
            )
            self.assertEqual(
                json.loads((checkpoint / "update-prompts.json").read_text()),
                {"step": 5, "promptIds": ["a"]},
            )
            self.assertEqual(env["trainer"].update_prompt_ids, [])

    def test_optimizer_schedule_scores_training_every_update_and_evaluation_at_epochs(self):
        train = [{"task_id": "t"}]
        val = [{"task_id": "v"}]
        jobs = namespace["evaluation_schedule"](range(1, 16), train, val, True, [5, 10, 15], True)
        self.assertEqual([step for step, split, _ in jobs if split == "train"], list(range(1, 16)))
        self.assertEqual([step for step, split, _ in jobs if split == "validation"], [5, 10, 15])

    def test_identity_adapter_requires_first_zero_rate_update_and_zero_lora_weights(self):
        import json, tempfile
        from types import SimpleNamespace
        from unittest.mock import patch, Mock

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            state = {"global_step": 1, "log_history": [{"step": 1, "learning_rate": 0.0}]}
            marker = path / "trainer_state.json"
            marker.write_text(json.dumps(state))
            tensor = SimpleNamespace(count_nonzero=lambda: SimpleNamespace(item=lambda: 0))
            loader = Mock(return_value={"model.lora_B.weight": tensor})
            namespace["json"] = json
            with patch.dict(
                "sys.modules", {"safetensors.torch": SimpleNamespace(load_file=loader)}
            ):
                self.assertTrue(namespace["zero_rate_warmup_checkpoint"](path))
                loader.return_value = {}
                self.assertFalse(namespace["zero_rate_warmup_checkpoint"](path))
                loader.return_value = {
                    "model.lora_B.weight": SimpleNamespace(
                        count_nonzero=lambda: SimpleNamespace(item=lambda: 1)
                    )
                }
                self.assertFalse(namespace["zero_rate_warmup_checkpoint"](path))
                state["global_step"] = 2
                marker.write_text(json.dumps(state))
                self.assertFalse(namespace["zero_rate_warmup_checkpoint"](path))
                state["global_step"] = 1
                state["log_history"][0]["learning_rate"] = 1e-4
                marker.write_text(json.dumps(state))
                self.assertFalse(namespace["zero_rate_warmup_checkpoint"](path))
            output = SimpleNamespace(token_ids=[1], logprobs=[{1: -0.1}])
            engine = SimpleNamespace(generate=lambda *a, **kw: [SimpleNamespace(outputs=[output])])
            request = SimpleNamespace(lora_name="step-1")
            with patch.dict(
                "sys.modules", {"vllm": SimpleNamespace(SamplingParams=lambda **kw: kw)}
            ):
                with self.assertRaisesRegex(RuntimeError, "no observable effect"):
                    namespace["check_adapter"](engine, request, "prompt", base=output)
                check = namespace["check_adapter"](
                    engine, request, "prompt", base=output, allow_identity=True
                )
                self.assertFalse(check["changed_tokens"])
                self.assertIn("zero learning rate", check["identity_reason"])

    def test_update_reward_uses_consumed_shuffled_prompts_and_partial_batch(self):
        import json, tempfile
        from types import SimpleNamespace

        source = ast.parse((Path(__file__).resolve().parents[1] / "sft.py").read_text())
        helper = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "training_prompt_ids"
        )
        env = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "sft.py", "exec"), env)
        tensor = lambda rows: SimpleNamespace(tolist=lambda: rows)
        index = {(4, 5): "b", (1, 2, 3): "a"}
        inputs = {
            "input_ids": tensor([[4, 5, 0, 0], [1, 2, 3, 0]]),
            "attention_mask": tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
        }
        self.assertEqual(env["training_prompt_ids"](inputs, index), ["b", "a"])
        with self.assertRaises(KeyError):
            env["training_prompt_ids"](inputs, {(1, 2, 3): "a"})
        train = [{"task_id": i} for i in ("a", "b", "c")]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint-5"
            checkpoint.mkdir()
            (checkpoint / "update-prompts.json").write_text(
                json.dumps({"step": 5, "promptIds": ["b"]})
            )
            namespace["json"] = json
            batch = namespace["optimizer_batch_tasks"](checkpoint, train)
            self.assertEqual(batch, [train[1]])
            rows = [
                {"promptId": id, "split": "train", "version": "step-5", "reward": reward}
                for id, reward in [("a", 1), ("b", 0), ("c", 1)]
            ]
            self.assertEqual(namespace["checkpoint_mean"](rows, batch, "train", "step-5"), 0)
            jobs = namespace["evaluation_schedule"](range(1, 6), train, [], True, [5], True, True)
            self.assertEqual(
                [(s, k) for s, k, _ in jobs],
                [
                    (1, "train"),
                    (2, "train"),
                    (3, "train"),
                    (4, "train"),
                    (5, "train"),
                    (5, "gallery"),
                    (5, "validation"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
