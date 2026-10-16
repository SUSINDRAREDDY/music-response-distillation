"""Export allowlisted records from the original run; never export account/session logs."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, project, output = args.source, args.project, args.output
    output.mkdir(parents=True, exist_ok=True)
    accepted = [json.loads(line) for line in (source/'data').read_text().splitlines()]
    settings = read(source/'settings.json')
    rejected = read(source/'collection/rejected.json')
    manifest = read(project/'data/manifest.json')
    tasks = {r['task_id']: r for line in (project/'data/train.jsonl').read_text().splitlines() for r in [json.loads(line)]}
    train_ids = {r['task_id'] for r in accepted}
    reject_ids = {r['task_id'] for r in rejected}
    assert len(train_ids) == len(accepted) == settings['train_examples'] == 607
    assert train_ids == set(settings['train_task_ids'])
    assert len(reject_ids) == len(rejected) == 199
    assert not train_ids & reject_ids
    assert train_ids | reject_ids == set(tasks)
    groups = {'train': [], 'rejected': []}
    accepted_by_id = {r['task_id']: r for r in accepted}
    for task_id in sorted(tasks):
        folder = source/'teacher/teacher/train'/task_id/'0'
        generation = read(folder/'generation.json')
        score = read(folder/'score.json')
        answer = (folder/'answer.txt').read_text()
        assert answer == generation['answer']
        used = task_id in train_ids
        assert score['accepted'] == used
        task = tasks[task_id]
        row = {
            'task_id': task_id, 'source_split': 'train',
            'prompt': task['prompt'], 'teacher_prompt': generation['prompt'],
            'teacher_answer': answer,
            'completion': accepted_by_id[task_id]['completion'] if used else [],
            'accepted': used, 'used_for_training': used,
            'reward': score['reward'], 'rejection_reason': score.get('reason') if not used else None,
            'bars_by_voice_json': json.dumps(score.get('bars_by_voice', {}), sort_keys=True),
            'diagnostics_json': json.dumps(score.get('diagnostics', {}), sort_keys=True),
            'teacher_model': generation['model'], 'reasoning_effort': generation['effort'],
            'generation_settings_json': json.dumps({'fresh_context': generation['fresh_context'], 'tool_calls': generation['tool_calls'], 'temperature': None, 'temperature_note': 'Not recorded; provider default'}, sort_keys=True),
            'attempt': 0, 'language': task['lang'], 'length': task['length'],
            'source_dataset': 'XiaomiMiMo/MiMo-V2.6-RL-oss', 'source_revision': manifest['revision'],
            'run_id': '49355152800c',
        }
        if used:
            assert row['reward'] == accepted_by_id[task_id]['reward']
            assert row['prompt'] == accepted_by_id[task_id]['prompt']
        groups['train' if used else 'rejected'].append(row)
    for split in ('validation', 'test'):
        rows = [json.loads(x) for x in (project/f'data/{split}.jsonl').read_text().splitlines()]
        ids = {r['task_id'] for r in rows}
        assert not ids & set(tasks)
        groups[split] = [{k: r[k] for k in ('task_id','prompt','lang','length','tag','bpm','meter','nvoice_want')} for r in rows]
    assert not {r['task_id'] for r in groups['validation']} & {r['task_id'] for r in groups['test']}
    for name, rows in groups.items():
        (output/f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    dump(output/'manifest.json', {
        'run_id': '49355152800c', 'counts': {k:len(v) for k,v in groups.items()},
        'source': manifest, 'checks': ['exact training-manifest IDs', 'unique IDs', 'disjoint source splits', '806 final outcomes covered', 'original answers preserved', 'training prompts and rewards match'],
        'sha256': {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(output.glob('*.jsonl'))},
    })
    print(json.dumps({k:len(v) for k,v in groups.items()}))


if __name__ == '__main__':
    main()
