import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import check_adapter, validate_adapter, validate_outputs  # noqa: E402


def output(tokens, logprobs=None):
    return SimpleNamespace(outputs=[SimpleNamespace(token_ids=tokens, logprobs=logprobs)])


@pytest.mark.parametrize("batch", [[], [output([1])], [output([1]), output([])]])
def test_incomplete_batch_cannot_publish_a_reward(batch):
    with pytest.raises(RuntimeError, match="incomplete batch"):
        validate_outputs(batch, 2)


def test_complete_batch_is_accepted():
    validate_outputs([output([1]), output([2, 3])], 2)


@pytest.mark.parametrize("adapted", [output([1], [-0.1]), output([2], [-0.2])])
def test_adapter_probe_rejects_silent_noop(monkeypatch, adapted):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    rows = iter([[output([1], [-0.1])], [adapted]])
    engine = SimpleNamespace(generate=lambda *a, **kw: next(rows))
    request = SimpleNamespace(lora_name="epoch-3")
    if adapted.outputs[0].token_ids == [1]:
        with pytest.raises(RuntimeError, match="no observable effect"):
            check_adapter(engine, request, "prompt")
    else:
        assert check_adapter(engine, request, "prompt")["changed_tokens"]


@pytest.mark.parametrize(
    "change", [{"base_model_name_or_path": "other"}, {"revision": "wrong"}, {"r": 8}]
)
def test_wrong_checkpoint_rejected_before_gpu_loading(tmp_path, change):
    config = {"model": "Qwen/test", "revision": "pinned", "sft": {"lora": {"r": 16}}}
    adapter = {"base_model_name_or_path": "Qwen/test", "revision": "pinned", "r": 16}
    (tmp_path / "adapter_config.json").write_text(json.dumps({**adapter, **change}))
    (tmp_path / "adapter_model.safetensors").touch()
    with pytest.raises(ValueError, match="differs"):
        validate_adapter(tmp_path, config)


def test_valid_adapter_and_missing_weights(tmp_path):
    config = {"model": "Qwen/test", "revision": "pinned", "sft": {"lora": {"r": 16}}}
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/test",
                "revision": None,
                "r": 16,
            }
        )
    )
    with pytest.raises(FileNotFoundError):
        validate_adapter(tmp_path, config)
    (tmp_path / "adapter_model.safetensors").touch()
    validate_adapter(tmp_path, config)


def test_shared_base_probe_still_checks_each_adapter(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs["lora_request"].lora_name)
        return [output([1], [-0.2])]

    engine = SimpleNamespace(generate=generate)
    base = output([1], [-0.1]).outputs[0]
    for name in ("epoch-1", "epoch-2", "epoch-3"):
        assert check_adapter(engine, SimpleNamespace(lora_name=name), "prompt", base)[
            "changed_logprobs"
        ]
    assert calls == ["epoch-1", "epoch-2", "epoch-3"]


def test_retry_replaces_only_matching_score():
    from inference import upsert_score

    original = dict(promptId="a", split="train", version="step-3", reward=0)
    other = {**original, "split": "validation"}
    manifest = {"verifierScores": [original, original.copy(), other]}
    replacement = {**original, "reward": 0.8}
    upsert_score(manifest, replacement)
    upsert_score(manifest, replacement)
    assert manifest["verifierScores"] == [other, replacement]


@pytest.mark.parametrize("cached", [True, False])
def test_prepare_requests_large_worker_only_on_cache_miss(tmp_path, monkeypatch, cached):
    import inference

    (tmp_path / "config.json").write_text("{}")
    if cached:
        (tmp_path / "export.json").write_text("{}")
    calls = []
    monkeypatch.setattr(inference, "ROOT", tmp_path)
    monkeypatch.setattr(inference, "text_model_path", lambda config: tmp_path)
    monkeypatch.setattr(inference, "cache", SimpleNamespace(reload=lambda: calls.append("reload")))
    monkeypatch.setattr(
        inference,
        "export_model",
        SimpleNamespace(remote=lambda: calls.append("export") or str(tmp_path)),
    )
    assert inference.prepare_model.local() == str(tmp_path)
    assert calls == (["reload"] if cached else ["reload", "export"])


def test_optimizer_evaluation_keeps_batch_rewards_and_epoch_galleries():
    from inference import evaluation_schedule

    train, validation = [{"task_id": "t"}], [{"task_id": "v"}]
    schedule = evaluation_schedule(
        tuple(range(1, 16)), train, validation, True, (5, 10, 15), True, True
    )
    assert [step for step, split, _ in schedule if split == "train"] == list(range(1, 16))
    assert [step for step, split, _ in schedule if split == "validation"] == [5, 10, 15]
    assert [step for step, split, _ in schedule if split == "gallery"] == [5, 10, 15]


def test_checkpoint_wait_aborts_when_training_has_failed(tmp_path, monkeypatch):
    import inference

    monkeypatch.setattr(inference, "results", SimpleNamespace(reload=lambda: None))
    (tmp_path / "status.json").write_text(json.dumps({"state": "failed"}))
    with pytest.raises(RuntimeError, match="Training failed"):
        inference.wait_for_file(tmp_path / "ready.json", tmp_path)


def test_epoch_baseline_schedule_compares_identical_sets_without_update_jobs():
    from inference import evaluation_schedule, pending_tasks, checkpoint_mean

    train = [{"task_id": "accepted-a"}, {"task_id": "accepted-b"}]
    validation = [{"task_id": "held-out"}]
    jobs = evaluation_schedule((1, 2, 3, 4, 5, 6), train, validation,
                               streaming=True, epoch_steps=(2, 4, 6), baseline=True)
    assert [(step, split) for step, split, _ in jobs] == [
        (0, "train"), (0, "validation"), (2, "train"), (2, "validation"),
        (4, "train"), (4, "validation"), (6, "train"), (6, "validation")]
    assert all(tasks == (train if split == "train" else validation)
               for _, split, tasks in jobs)
    scores = [{"promptId": "accepted-a", "split": "train", "version": "step-0", "reward": 0}]
    assert pending_tasks({"verifierScores": scores}, train, "train", "step-0") == train[1:]
    with pytest.raises(RuntimeError, match="Incomplete"):
        checkpoint_mean(scores, train, "train", "step-0")
    scores.append({"promptId": "accepted-b", "split": "train", "version": "step-0", "reward": 1})
    assert checkpoint_mean(scores, train, "train", "step-0") == 0.5


def test_windowed_throughput_handles_idle_and_batch_reset():
    from inference import throughput_sample
    value, previous = throughput_sample({'batchId': 1, 'state': 'generating', 'outputTokens': 100, 'elapsedSeconds': 2}, None, 10)
    assert value['outputTokensPerSecond'] == 50
    value, previous = throughput_sample({'batchId': 1, 'state': 'generating', 'outputTokens': 140}, previous, 12)
    assert value['outputTokensPerSecond'] == 20
    value, previous = throughput_sample({'batchId': 1, 'state': 'scoring', 'outputTokens': 140, 'batchAverageTokensPerSecond': 35}, previous, 14)
    assert value['outputTokensPerSecond'] == 0
    assert value['batchAverageTokensPerSecond'] == 35
    value, previous = throughput_sample({'batchId': 2, 'state': 'generating', 'outputTokens': 30, 'elapsedSeconds': 1}, previous, 16)
    assert value['outputTokensPerSecond'] == 30
    value, _ = throughput_sample({'state': 'waiting'}, previous, 18)
    assert value['outputTokensPerSecond'] == 0


def test_grouped_checkpoint_generation_keeps_splits_and_failed_denominators(tmp_path, monkeypatch):
    import inference
    import inference_engine
    import recordings
    import reward
    real_path = Path
    root = tmp_path / 'runs'
    output_dir = root / 'sample'
    output_dir.mkdir(parents=True)
    public = root / 'public/runs/test'
    public.mkdir(parents=True)
    config = json.loads((inference.ROOT / 'config.json').read_text())
    (tmp_path / 'config.json').write_text(json.dumps(config))
    monkeypatch.setattr(inference, 'ROOT', tmp_path)
    monkeypatch.setattr(inference, 'Path', lambda p: root / str(p).removeprefix('/runs').lstrip('/') if str(p).startswith('/runs') else real_path(p))
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    monkeypatch.setattr(inference, 'results', volume)
    monkeypatch.setattr(inference, 'cache', volume)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'export.json').write_text('{}')
    monkeypatch.setattr(inference, 'text_model_path', lambda _: model)
    def task(identity):
        return {'task_id': identity, 'prompt': [{'content': identity}]}
    train = [task('train-a'), task('train-b')]
    validation = [task('eval-a')]
    plan = {'run_name':'sample','run_id':'test','epochs':1,'train_tasks':train,'validation_tasks':validation,
            'stream_evaluation':True,'training_reward_scope':'fixed_training_set','baseline_generation':True,
            'generation':{'do_sample':False,'max_new_tokens':8192,'seed':42},'scorer_revision':'pinned'}
    (output_dir / 'plan.json').write_text(json.dumps(plan))
    (output_dir / 'settings.json').write_text(json.dumps({**config, 'train_examples':2,'train_task_ids':['train-a','train-b'],'settings':config['sft']['trainer']}))
    (output_dir / 'status.json').write_text('{"state":"training"}')
    checkpoint = output_dir / 'checkpoint-1'
    checkpoint.mkdir()
    (checkpoint / 'ready.json').write_text('{}')
    (public / 'evaluation.json').write_text('{"verifierScores":[]}')
    monkeypatch.setattr(inference, 'validate_adapter', lambda *a: None)
    monkeypatch.setattr(inference, 'zero_rate_warmup_checkpoint', lambda *a: False)
    monkeypatch.setattr(inference, 'check_adapter', lambda *a, **kw: {'changed_tokens':True})
    monkeypatch.setattr(reward, 'check_scorer', lambda: None)
    monkeypatch.setitem(sys.modules,'torch', SimpleNamespace(__version__='test',set_num_threads=lambda n:None,cuda=SimpleNamespace(get_device_name=lambda n:'H100')))
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages,**kw:messages[0]['content'],convert_tokens_to_ids=lambda _:1)
    monkeypatch.setitem(sys.modules,'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:tokenizer)))
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(__version__='test',SamplingParams=lambda **kw:kw))
    monkeypatch.setitem(sys.modules,'vllm.lora.request',SimpleNamespace(LoRARequest=lambda *args:args))
    class Run:
        config = {}
        summary = {}
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def log(self,*a): pass
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=lambda **kw:Run()))
    calls = []
    class Engine:
        def __init__(self,**kw):
            assert kw['max_num_seqs'] == 3
            self.telemetry = {}
        def generate(self,*a,**kw): return [output([1])]
        def completed(self,prompts,*a):
            calls.append(prompts)
            for i in reversed(range(len(prompts))):
                yield i, SimpleNamespace(outputs=[SimpleNamespace(text=prompts[i],token_ids=[1])]), {'tokens':1,'request_seconds':1,'abort_reason':None,'finish_reason':'stop'}
    monkeypatch.setattr(inference_engine,'ContinuousEngine',Engine)
    def record(task,*a,**kw):
        # Held-out failure still contributes zero to its own fixed denominator.
        return None, {'accepted': task['task_id'] != 'eval-a','reward':0 if task['task_id']=='eval-a' else .5}
    monkeypatch.setattr(recordings,'public_recording',record)
    inference.main_job(run_name='sample')
    assert calls == [['train-a','train-b','eval-a']]*2
    manifest=json.loads((public/'evaluation.json').read_text())
    assert [(x['step'],x['sampleCount'],x['reward']) for x in manifest['trainingCheckpoints']] == [(0,2,.5),(1,2,.5)]
    assert [(x['step'],x['sampleCount'],x['reward']) for x in manifest['checkpoints']] == [(0,1,0),(1,1,0)]
    assert len(manifest['verifierScores']) == 6
    # Resume must not regenerate already-scored failures or duplicate outputs.
    inference.main_job(run_name='sample')
    assert calls[-2:] == [[],[]]
    assert len(json.loads((public/'evaluation.json').read_text())['verifierScores']) == 6
