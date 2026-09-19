# [model-review] - START
"""Model-review evidence and reports. These results are never human ground truth."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import html
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

FAMILIES = {'maximum_speed', 'cancellation', 'zone', 'variable_display',
            'road_marking', 'other_sign', 'not_a_sign', 'unreadable'}
NUMERIC = {'maximum_speed', 'cancellation', 'zone', 'variable_display', 'road_marking'}
UNCERTAIN = {'AMBIGUOUS', 'UNREADABLE', 'LOW_CONFIDENCE', 'DISAGREEMENT',
             'UNAVAILABLE', 'STILL_AMBIGUOUS'}
RESOLUTIONS = {'OK': 'RESOLVED_OK', 'FALSE_POSITIVE': 'RESOLVED_FALSE_POSITIVE',
               'FALSE_NEGATIVE': 'RESOLVED_FALSE_NEGATIVE', 'WRONG_VALUE': 'RESOLVED_WRONG_VALUE',
               'APPLICABILITY_ERROR': 'RESOLVED_OTHER_ROAD'}
RESULTS = UNCERTAIN | set(RESOLUTIONS) | set(RESOLUTIONS.values()) | {'WRONG_SIGN_FAMILY'}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def score(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_review(review: dict) -> dict:
    fields = {'detected', 'value_kph', 'sign_family', 'confidence', 'applicability'}
    require(isinstance(review, dict) and fields <= review.keys(), 'missing_review_fields')
    require(type(review['detected']) is bool, 'invalid_detected')
    family, value = review['sign_family'], review['value_kph']
    require(isinstance(family, str) and family in FAMILIES, 'invalid_sign_family')
    require(value is None or (type(value) is int and 1 <= value <= 200), 'invalid_value')
    require(value is None or family in NUMERIC, 'value_family_mismatch')
    require(score(review['confidence']), 'invalid_confidence')
    require(review['applicability'] in ('own_road', 'other_road', 'unresolved'), 'invalid_applicability')
    if not review['detected']:
        require(value is None and family == 'not_a_sign', 'contradictory_absence')
    else:
        require(family != 'not_a_sign', 'contradictory_detection')
    return dict(review)


def compare_review(predictions: list[dict], review: dict, threshold: float = .9) -> str:
    answer = validate_review(review)
    require(score(threshold), 'invalid_threshold')
    require(isinstance(predictions, list), 'missing_predictions')
    normalized = []
    for prediction in predictions:
        required = {'family', 'value_kph', 'score', 'bbox_xyxy'}
        require(isinstance(prediction, dict) and required <= prediction.keys(), 'invalid_prediction')
        family = prediction['family']
        family = 'road_marking' if family == 'road_marking_candidate' else family
        value, box = prediction['value_kph'], prediction['bbox_xyxy']
        require(isinstance(family, str) and family in FAMILIES, 'invalid_prediction_family')
        require(value is None or (type(value) is int and 1 <= value <= 200 and family in NUMERIC), 'invalid_prediction_value')
        require(score(prediction['score']), 'invalid_prediction_score')
        require(isinstance(box, list) and len(box) == 4
                and all(type(v) in (int, float) and math.isfinite(v) for v in box)
                and 0 <= box[0] < box[2] and 0 <= box[1] < box[3], 'invalid_prediction_bbox')
        normalized.append({**prediction, 'family': family})
    if len(normalized) > 1:
        return 'AMBIGUOUS'
    if answer['sign_family'] == 'unreadable' or (answer['detected'] and answer['sign_family'] in NUMERIC and answer['value_kph'] is None):
        return 'UNREADABLE'
    if answer['confidence'] < threshold:
        return 'LOW_CONFIDENCE'
    if answer['detected'] and answer['applicability'] == 'other_road':
        # A detector with no road-ownership output has not asserted OWN.
        return 'APPLICABILITY_ERROR' if normalized and normalized[0].get('applicability') == 'own_road' else 'AMBIGUOUS'
    if not normalized:
        return 'FALSE_NEGATIVE' if answer['detected'] else 'OK'
    if not answer['detected']:
        return 'FALSE_POSITIVE'
    if normalized[0]['family'] != answer['sign_family']:
        return 'WRONG_SIGN_FAMILY'
    if normalized[0]['value_kph'] != answer['value_kph']:
        return 'WRONG_VALUE'
    return 'OK'


def second_pass_result(first_result: str, second_result: str) -> str:
    require(first_result in UNCERTAIN, 'first_pass_not_ambiguous')
    require(second_result in RESULTS, 'invalid_second_result')
    return RESOLUTIONS.get(second_result, 'STILL_AMBIGUOUS')


def _validate_record(row: dict) -> None:
    required = {'image', 'image_sha256', 'route_id', 'segment_index', 'frame_index', 'label_basis',
                'reviewer', 'review_timestamp', 'review_model', 'review_model_version',
                'review_model_sha256', 'result'}
    require(isinstance(row, dict) and required <= row.keys(), 'missing_audit_provenance')
    require(row['result'] in RESULTS, 'invalid_audit_result')
    require(row['label_basis'] == 'model_review', 'not_model_review')
    unavailable = row['result'] == 'UNAVAILABLE' and isinstance(row.get('reason'), str) and bool(row['reason'])
    for field in ('image', 'route_id', 'review_timestamp'):
        require(isinstance(row[field], str) and bool(row[field].strip()), 'invalid_' + field)
    require(type(row['segment_index']) is int and row['segment_index'] >= 0, 'invalid_segment_index')
    require(row['frame_index'] is None or (type(row['frame_index']) is int and row['frame_index'] >= 0), 'invalid_frame_index')
    if 'output_index' in row:
        require(type(row['output_index']) is int and row['output_index'] >= 0, 'invalid_output_index')
    for field in ('image_sha256', 'review_model_sha256'):
        value = row[field]
        require((unavailable and value is None) or (isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)), 'invalid_' + field)
    for field in ('review_model', 'review_model_version'):
        value = row[field]
        require((unavailable and value is None) or (isinstance(value, str) and bool(value.strip()) and value not in ('main', 'master', 'latest')), 'invalid_' + field)
    require((unavailable and row['review_model'] is None and row['reviewer'] is None)
            or row['reviewer'] == 'codex/' + str(row['review_model']), 'invalid_reviewer')
    stamp = datetime.fromisoformat(row['review_timestamp'].replace('Z', '+00:00'))
    require(stamp.tzinfo is not None and stamp.utcoffset() is not None, 'invalid_review_timestamp')


def write_report(rows: list[dict], output: Path) -> dict:
    """Publish one complete immutable report directory, or publish nothing."""
    output = Path(output)
    require(isinstance(rows, list), 'invalid_audit_rows')
    identities = set()
    for row in rows:
        _validate_record(row)
        identity = (row['route_id'], row['segment_index'], row['frame_index'], row.get('output_index'), row['image'])
        require(identity not in identities, 'duplicate_audit_frame')
        identities.add(identity)
    payload = ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows)
    if output.exists():
        raise FileExistsError(output)
    counts = dict(sorted(Counter(row['result'] for row in rows).items()))
    summary = {'schema_version': 1, 'label_basis': 'model_review', 'ground_truth': False,
               'frame_count': len(rows), 'counts': counts}
    categories = {'ambiguous': UNCERTAIN,
                  'false_positives': {'FALSE_POSITIVE', 'RESOLVED_FALSE_POSITIVE'},
                  'false_negatives': {'FALSE_NEGATIVE', 'RESOLVED_FALSE_NEGATIVE'},
                  'wrong_values': {'WRONG_VALUE', 'RESOLVED_WRONG_VALUE'},
                  'applicability_errors': {'APPLICABILITY_ERROR', 'RESOLVED_OTHER_ROAD'},
                  'wrong_sign_families': {'WRONG_SIGN_FAMILY'}}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.review-', dir=output.parent))
    try:
        (staging/'audit.jsonl').write_text(payload, encoding='utf-8')
        (staging/'run.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        for name, results in categories.items():
            directory = staging/name
            directory.mkdir()
            directory.joinpath('records.jsonl').write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in rows if r['result'] in results), encoding='utf-8')
        body = ''.join('<tr><td>' + html.escape(row['image']) + '</td><td>' + html.escape(row['result'])
                       + '</td><td><pre>' + html.escape(json.dumps(row, sort_keys=True, ensure_ascii=False)) + '</pre></td></tr>' for row in rows)
        document = '<!doctype html><meta charset="utf-8"><title>NNSLR model review</title><h1>NNSLR model review</h1><p>Automatic evidence; not human ground truth.</p><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre><table><tr><th>Image</th><th>Result</th><th>Evidence</th></tr>' + body + '</table>'
        (staging/'report.html').write_text(document, encoding='utf-8')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary
# [model-review] - END
