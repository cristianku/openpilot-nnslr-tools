# [model-review] - START
import copy
import json
import pytest
from test_annotations import reviewed, write, invoke
from nnslr_tools.annotations import load_jsonl, validate_dataset


def model_frame(root, route='00000001--abcdef1234', **kwargs):
    row = reviewed(root, route, **kwargs)
    row.update(label_basis='model_review', reviewer='codex/test-visual-model',
               review_model='test-visual-model', review_model_version='revision-123',
               review_model_sha256='a' * 64, confidence=.95)
    return row


def import_model(root, capsys, rows):
    code, out = invoke(capsys, 'import-annotations', str(write(root, rows)),
                       '--data-root', str(root), '--dataset-kind', 'training-candidate')
    assert code == 0, out.out + out.err
    return root / 'datasets/training_candidate_dataset/objects.jsonl'


def test_candidate_default_is_separate_and_never_human(tmp_path, capsys):
    path = import_model(tmp_path, capsys, [model_frame(tmp_path, negative=True)])
    row = load_jsonl(path)[0]
    assert not (tmp_path / 'annotations/objects.jsonl').exists()
    assert row['schema_version'] == 2
    assert row['proposal_source'] == 'model_review'
    assert row['provenance']['label_basis'] == 'model_review'
    assert row['provenance']['review_model_sha256'] == 'a' * 64
    assert row['provenance']['confidence'] == .95
    assert row['decoded_frame_index'] is None and row['capture_mono_ns'] is None
    code, out = invoke(capsys, 'validate-dataset', '--data-root', str(tmp_path), '--dataset-kind', 'training-candidate')
    assert code == 0 and json.loads(out.out)['valid']
    assert not validate_dataset([row], tmp_path)['valid']
    code, out = invoke(capsys, 'import-annotations', str(path), '--data-root', str(tmp_path))
    assert code == 1 and 'model_review_not_gold' in out.out
    assert not (tmp_path / 'annotations/objects.jsonl').exists()


@pytest.mark.parametrize('field,value', [
    ('review_model', None), ('review_model_version', ''), ('review_model_version', 'latest'),
    ('review_model_sha256', 'bad'), ('confidence', True), ('confidence', -1),
    ('confidence', 1.1), ('reviewer', 'human'), ('review_timestamp', '2026-09-19'),
    ('review_state', 'pending'), ('review_state', 'unresolved'), ('review_state', 'unavailable'),
    ('review_state', 'resolved'), ('frame_sha256', '0' * 64),
])
def test_invalid_model_reviews_do_not_publish(tmp_path, capsys, field, value):
    row = model_frame(tmp_path); row[field] = value
    code, out = invoke(capsys, 'import-annotations', str(write(tmp_path, [row])),
                       '--data-root', str(tmp_path), '--dataset-kind', 'training-candidate')
    assert code == 1, out.out + out.err
    assert not (tmp_path / 'datasets/training_candidate_dataset/objects.jsonl').exists()


def test_candidate_split_kind_cannot_be_relabelled_gold(tmp_path, capsys):
    frames = [model_frame(tmp_path, f'{i+1:08x}--abcdef1234', negative=i == 0) for i in range(8)]
    path = import_model(tmp_path, capsys, frames)
    code, out = invoke(capsys, 'build-splits', '--data-root', str(tmp_path), '--dataset-kind', 'training-candidate')
    assert code == 0, out.out + out.err
    result = json.loads(out.out)
    split_path = tmp_path / result['split_path']
    split = json.loads(split_path.read_text())
    assert split['dataset_kind'] == 'training-candidate'
    assert not (tmp_path / 'splits/latest.json').exists()
    assert set(split['assignments'].values()) == {'train', 'test', 'validation', 'route-held-out', 'hard-negative'}
    split['dataset_kind'] = 'gold'; split_path.write_text(json.dumps(split))
    code, out = invoke(capsys, 'validate-dataset', str(path), '--data-root', str(tmp_path),
                       '--dataset-kind', 'training-candidate', '--splits', str(split_path))
    assert code == 1 and 'split_kind_mismatch' in out.out


def test_candidate_keeps_v1_human_rows_and_requires_five_groups(tmp_path, capsys):
    path = import_model(tmp_path, capsys, [reviewed(tmp_path), model_frame(tmp_path, '00000002--abcdef1234')])
    assert {r['schema_version'] for r in load_jsonl(path)} == {1, 2}
    code, out = invoke(capsys, 'build-splits', '--data-root', str(tmp_path), '--dataset-kind', 'training-candidate')
    assert code == 1 and 'insufficient_independent_groups' in out.out


def test_schema_versions_and_model_hash_patterns(tmp_path, capsys):
    from pathlib import Path
    import re
    schema = json.loads((Path(__file__).resolve().parents[2] / 'schemas/annotation.schema.json').read_text())
    path = import_model(tmp_path, capsys, [model_frame(tmp_path)])
    row = load_jsonl(path)[0]
    pattern = schema['properties']['provenance']['properties']['review_model_sha256']['pattern']
    assert re.fullmatch(pattern, row['provenance']['review_model_sha256'])
    assert not re.fullmatch(pattern, 'bad')
    row['schema_version'] = 1
    assert not validate_dataset([row], tmp_path, dataset_kind='training-candidate')['valid']


# [model-review] - START
import argparse
import json as _json
import hashlib as _hashlib
from test_annotations import png
from nnslr_tools import model_review as _mr


def _negative_review():
    return dict(detected=False, value_kph=None, sign_family='not_a_sign', confidence=.95,
                applicability='unresolved', multiple_visible=False, proposal_matches_review=False)


def _negative_source(root, i):
    route = f'{i + 1:08x}--abcdef1234'
    image = root / 'derived/frames' / route / '0' / 'frame_00000000.png'
    png(image, (i + 1) * 7)  # distinct bytes per route -> distinct image groups
    return dict(route_id=route, segment_index=0, output_index=0,
                image_path=str(image.relative_to(root)),
                frame_sha256=_hashlib.sha256(image.read_bytes()).hexdigest(),
                camera_stream='narrow_road', decoded_frame_index=None, detections=[],
                model={'sha256': 'b' * 64}, width=32, height=24,
                alignment_status='unresolved', capture_mono_ns=None, capture_time_provenance='unknown')


def test_publish_repairs_candidate_split_pointer_and_resume_summary(tmp_path, monkeypatch):
    # >=5 independent routes, all confirmed negatives -> build_splits succeeds.
    sources = [_negative_source(tmp_path, i) for i in range(5)]
    manifest = tmp_path / 'preannotations.jsonl'
    manifest.write_text(''.join(_json.dumps(s) + '\n' for s in sources))
    meta = tmp_path / 'model.json'; meta.write_text(_json.dumps({'model_id': 'm', 'revision': 'r1', 'model_sha256': 'a' * 64}))
    server = tmp_path / 'server.json'; server.write_text(_json.dumps({'port': 18089}))
    output = tmp_path / 'review'
    args = argparse.Namespace(model_metadata=meta, server_metadata=server, preannotations=[manifest], limit=0)
    monkeypatch.setattr(_mr, 'verify_local_model', lambda *_: None)
    monkeypatch.setattr(_mr.subprocess, 'check_output', lambda *a, **k: 'test-commit\n')
    monkeypatch.setattr(_mr, 'LocalVisualReviewer', lambda _port: lambda _p, _m, _b: {'review': _negative_review(), 'raw_response': 'synthetic'})

    assert _mr.run(args, tmp_path, output) == 0
    final = output / 'final'
    assert (final / 'candidate_splits.json').is_file()
    assert (final / 'summary.json').is_file()

    pointer_path = tmp_path / 'datasets/training_candidate_dataset/splits/latest.json'
    assert pointer_path.is_file(), 'durable candidate split pointer was not published'
    pointer = _json.loads(pointer_path.read_text())
    target = tmp_path / pointer['path']
    assert target.is_file(), f'latest.json points at a missing path: {pointer["path"]}'
    assert '.publish' not in pointer['path'], 'latest.json still points at a staging path'
    assert target.read_bytes() == (final / 'candidate_splits.json').read_bytes()
    assert pointer['sha256'] == _json.loads((final / 'splits.json').read_text())['split_sha256']

    # Resume a completed run: the summary must keep dataset_valid/candidate_frame_count.
    assert _mr.run(args, tmp_path, output) == 0
    status = _json.loads((output / 'run.json').read_text())
    assert status['status'] == 'completed'
    assert status['summary']['dataset_valid'] is True
    assert status['summary']['candidate_frame_count'] == 5
    assert status['summary']['splits_valid'] is True
# [model-review] - END
