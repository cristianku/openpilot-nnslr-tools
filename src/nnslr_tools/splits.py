# [reviewed-dataset] - START
"""Deterministic route/site/encounter/image-connected partitions, never frame randomization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nnslr_tools.annotations import DatasetError, atomic_write, require, validate_dataset

SPLITS = ('train', 'validation', 'test', 'route-held-out', 'hard-negative')


def connected_groups(rows: list[dict]) -> list[list[dict]]:
    active = [r for r in rows if r['review_state'] in ('accepted', 'corrected')]
    parent = list(range(len(active)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tokens = {}
    for i, row in enumerate(active):
        for field in ('route_id', 'site_group', 'encounter_id', 'image_sha256'):
            value = row[field]
            if value is None:
                continue
            key = field, value
            if key in tokens:
                parent[find(i)] = find(tokens[key])
            else:
                tokens[key] = i
    groups = {}
    for i, row in enumerate(active):
        groups.setdefault(find(i), []).append(row)
    return [sorted(group, key=lambda r: r['annotation_id']) for group in groups.values()]


def validate_splits(rows: list[dict], split: dict, dataset_sha256: str, *, dataset_kind: str = 'gold') -> list[dict]:
    errors = []
    if not isinstance(split, dict) or split.get('schema_version') != 1 or not isinstance(split.get('assignments'), dict):
        return [{'reason': 'invalid_split_schema'}]
    assignments = split['assignments']
    # [model-review] - START
    split_kind = split.get('dataset_kind', 'gold')
    if split_kind not in ('gold', 'training-candidate'):
        return [{'reason': 'unknown_dataset_kind', 'detail': str(split_kind)}]
    if split_kind != dataset_kind:
        return [{'reason': 'split_kind_mismatch', 'detail': 'split and dataset purposes differ'}]
    model_rows = [r for r in rows if isinstance(r.get('provenance'), dict) and r['provenance'].get('label_basis') == 'model_review']
    if split_kind == 'gold' and model_rows:
        return [{'reason': 'split_kind_mismatch', 'detail': 'gold split cannot cover model-reviewed rows'}]
    # [model-review] - END
    expected = {r['annotation_id'] for r in rows if r['review_state'] in ('accepted','corrected')}
    if set(assignments) != expected or any(not isinstance(s,str) or s not in SPLITS for s in assignments.values()):
        return [{'reason': 'invalid_split_membership'}]
    if split.get('dataset_sha256') != dataset_sha256:
        errors.append({'reason': 'split_dataset_mismatch'})
    for group in connected_groups(rows):
        assigned = {assignments.get(r['annotation_id']) for r in group if r['annotation_id'] in assignments}
        if len(assigned) > 1:
            errors.append({'reason': 'split_leakage', 'routes': sorted({r['route_id'] for r in group})})
    return errors


def build_splits(rows: list[dict], root: Path, *, seed: int = 0, output: Path | None = None,
                 dataset_kind: str = 'gold', publish_latest: bool = True) -> dict:
    # [model-review] - START
    report = validate_dataset(rows, root, dataset_kind=dataset_kind)
    # [model-review] - END
    require(report['valid'], 'invalid_dataset', json.dumps(report['errors']))
    groups = connected_groups(rows)
    require(len(groups) >= len(SPLITS), 'insufficient_independent_groups',
            f'{len(groups)} groups; five disjoint partitions require at least five. Keep connected routes/sites together.')
    require(type(seed) is int, 'invalid_seed')

    def rank(group):
        ids = ','.join(r['annotation_id'] for r in group)
        return hashlib.sha256(f'{seed}:{ids}'.encode()).hexdigest()

    groups.sort(key=rank)
    negative = next((g for g in groups if any(r['sign_family'] == 'not_a_sign' for r in g)), None)
    require(negative is not None, 'missing_hard_negatives', 'review sign-free frames or misleading objects first')
    groups.remove(negative)
    assignments = {r['annotation_id']: 'hard-negative' for r in negative}
    held_out = groups.pop(0)
    assignments.update({r['annotation_id']: 'route-held-out' for r in held_out})
    evaluation_count = max(1, round(len(groups) * .15))
    for index, group in enumerate(groups):
        split = 'validation' if index < evaluation_count else 'test' if index < evaluation_count * 2 else 'train'
        assignments.update({r['annotation_id']: split for r in group})
    # [model-review] - START
    base = root / 'datasets' / 'training_candidate_dataset' if dataset_kind == 'training-candidate' else root
    payload = {'schema_version': 1, 'dataset_kind': dataset_kind, 'dataset_sha256': report['dataset_sha256'], 'seed': seed,
    # [model-review] - END
               'grouping': ['route_id', 'site_group', 'encounter_id', 'image_sha256'],
               'assignments': dict(sorted(assignments.items())),
               'excluded_rejected_ids': sorted(r['annotation_id'] for r in rows if r['review_state'] == 'rejected'),
               'counts': {name: sum(s == name for s in assignments.values()) for name in SPLITS},
               'warnings': report['warnings'], 'leakage_checked': True,
               'hard_negative_policy': 'reserve complete connected groups containing reviewed negatives; positive controls stay in the same partition'}
    errors = validate_splits(rows, payload, report['dataset_sha256'], dataset_kind=dataset_kind)
    require(not errors, 'split_leakage', json.dumps(errors))
    content = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    digest = hashlib.sha256(content).hexdigest()
    # [model-review] - START
    target = output if output is not None else base / 'splits' / (digest + '.json')
    # [model-review] - END
    if target.exists():
        require(target.read_bytes() == content, 'immutable_split_conflict', str(target))
    else:
        atomic_write(target, content)
    # [model-review] - START
    # A staged split (e.g. the model-review runner's rename-to-final staging dir)
    # must not repoint the durable pointer at a path that is about to disappear;
    # the caller publishes the pointer once the split is durably in place.
    if publish_latest:
        atomic_write(base / 'splits/latest.json', (json.dumps({'path': str(target.relative_to(root)) if target.is_relative_to(root) else str(target),
                                                           'sha256': digest}, sort_keys=True) + '\n').encode())
    # [model-review] - END
    return {'valid': True, 'dataset_kind': dataset_kind, 'dataset_sha256': report['dataset_sha256'], 'split_sha256': digest,
            'split_path': str(target.relative_to(root)) if target.is_relative_to(root) else str(target),
            'counts': payload['counts'], 'warnings': report['warnings']}
# [reviewed-dataset] - END
