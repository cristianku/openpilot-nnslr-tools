# [reviewed-dataset] - START
"""Canonical reviewed annotations. Stdlib only; unknown timing stays unknown."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from nnslr_tools.manifest import normalize_relpath, parse_route_id, sha256_of
from nnslr_tools.media import MediaToolError, png_dimensions
from speed_vision_core.types import MIN_VALUE_KPH, MAX_VALUE_KPH

SCHEMA_VERSION = 1
# [model-review] - START
MODEL_SCHEMA_VERSION = 2
DATASET_KINDS = ('gold', 'training-candidate')
# [model-review] - END
ENUMS = {
    "sign_family": {"maximum_speed", "cancellation", "zone", "variable_display", "other_sign", "not_a_sign", "unreadable", "road_marking"},
    "value_state": {"read", "unreadable", "unknown", "not_applicable"},
    "applicability": {"own_road", "other_road", "unresolved"},
    "supplementary_panel": {"none", "present", "unresolved", "unknown"},
    "visibility": {"clear", "partial", "poor", "unknown"},
    "occlusion": {"none", "partial", "heavy", "unknown"},
    "review_state": {"pending", "accepted", "corrected", "rejected"},
    "camera_stream": {"narrow_road", "wide_road", "q_narrow_road"},
    "alignment_status": {"exact", "unresolved"},
    "capture_time_provenance": {"encode_index_sof", "encode_index_eof", "unknown"},
}
FIELDS = set(ENUMS) | {
    "schema_version", "annotation_id", "route_id", "segment_index", "output_index", "decoded_frame_index",
    "capture_mono_ns", "image_path", "image_sha256", "width", "height", "bbox_xyxy", "value_kph",
    "temporary", "variable_display", "proposal_source", "proposal_score", "reviewer", "review_timestamp", "notes",
    "source_video_sha256", "source_log_sha256", "provenance", "site_group", "encounter_id",
}


class DatasetError(ValueError):
    def __init__(self, reason: str, detail: str = ""):
        self.reason, self.detail = reason, detail
        super().__init__(f"{reason}: {detail}")


def require(condition: bool, reason: str, detail: str = "") -> None:
    if not condition:
        raise DatasetError(reason, detail)


def _nonfinite(value):
    raise DatasetError("invalid_json", f"non-finite token {value}")


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "invalid_json", f"duplicate key {key}")
        result[key] = value
    return result


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_nonfinite, object_pairs_hook=_object)
        except (ValueError, TypeError) as exc:
            raise DatasetError("invalid_json", f"{path.name}:{number}: {exc}") from exc
        require(isinstance(row, dict), "invalid_schema", f"line {number} must be an object")
        rows.append(row)
    return rows


def jsonl_bytes(rows: list[dict]) -> bytes:
    return ''.join(json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n' for r in rows).encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(payload)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _integer(value, minimum=0) -> bool:
    return type(value) is int and value >= minimum


def frame_identity(row: dict) -> tuple:
    # output_index identifies legacy image-only extraction; it never becomes a decoded index.
    index = ('decoded', row['decoded_frame_index']) if row['decoded_frame_index'] is not None else ('output', row['output_index'])
    return row['route_id'], row['segment_index'], row['camera_stream'], index


def validate_annotation(row: dict, root: Path, *, dataset_kind: str = 'gold') -> None:
    # [model-review] - START
    require(dataset_kind in DATASET_KINDS, 'unknown_dataset_kind', str(dataset_kind))
    # [model-review] - END
    require(set(row) == FIELDS, "invalid_schema", f"missing={sorted(FIELDS-set(row))}, unknown={sorted(set(row)-FIELDS)}")
    # [model-review] - START
    model = isinstance(row.get('provenance'), dict) and row['provenance'].get('label_basis') == 'model_review'
    if model:
        require(dataset_kind == 'training-candidate', 'model_review_not_gold', 'model review is training-candidate only')
        require(type(row['schema_version']) is int and row['schema_version'] == MODEL_SCHEMA_VERSION, 'invalid_schema_version')
    else:
        require(type(row['schema_version']) is int and row['schema_version'] == SCHEMA_VERSION, 'invalid_schema_version')
    # [model-review] - END
    for field, values in ENUMS.items():
        require(isinstance(row[field], str) and row[field] in values, "unknown_enum", field)
    require(row['review_state'] != 'pending', "unreviewed_proposal")
    for field in ('annotation_id', 'proposal_source', 'reviewer', 'review_timestamp'):
        require(isinstance(row[field], str) and bool(row[field].strip()), "missing_provenance", field)
    try:
        timestamp = datetime.fromisoformat(row['review_timestamp'].replace('Z', '+00:00'))
        require(timestamp.tzinfo is not None and timestamp.utcoffset() is not None, "invalid_review_timestamp")
    except ValueError as exc:
        raise DatasetError("invalid_review_timestamp", row['review_timestamp']) from exc
    require(isinstance(row['route_id'], str), "invalid_identity", "route_id")
    parse_route_id(row['route_id'])
    for field in ('segment_index', 'output_index'):
        require(_integer(row[field]), "invalid_identity", field)
    require(row['decoded_frame_index'] is None or _integer(row['decoded_frame_index']), "invalid_identity", "decoded_frame_index")
    for field in ('width', 'height'):
        require(_integer(row[field], 1), "invalid_dimensions", field)
    for field in ('temporary', 'variable_display'):
        require(row[field] is None or type(row[field]) is bool, "invalid_schema", field)
    for field in ('site_group', 'encounter_id'):
        require(row[field] is None or (isinstance(row[field], str) and bool(row[field].strip()) and row[field] != 'unknown'), "invalid_group", field)
    require(isinstance(row['notes'], str), "invalid_schema", "notes")
    require(_hash(row['image_sha256']), "missing_provenance", "image_sha256")
    for field in ('source_video_sha256', 'source_log_sha256'):
        require(row[field] is None or _hash(row[field]), "missing_provenance", field)
    # [model-review] - START
    provenance = row['provenance']
    if model:
        require(isinstance(provenance, dict) and _hash(provenance.get('source_review_sha256')),
                "missing_provenance", "review source")
        for field in ('review_model', 'review_model_version'):
            require(isinstance(provenance.get(field), str) and bool(provenance[field].strip()),
                    "missing_provenance", field)
        require(_hash(provenance.get('review_model_sha256')), "missing_provenance", "review_model_sha256")
        confidence = provenance.get('confidence')
        require(type(confidence) in (int, float) and not isinstance(confidence, bool) and math.isfinite(confidence)
                and 0 <= confidence <= 1, "invalid_confidence", str(confidence))
        require(row['reviewer'] == 'codex/' + provenance['review_model'], "reviewer_model_mismatch", row['reviewer'])
    else:
        require(isinstance(provenance, dict) and _hash(provenance.get('source_review_sha256'))
                and provenance.get('label_basis') == 'human_review', "missing_provenance", "review source")
    # [model-review] - END
    require(row['decoded_frame_index'] is None or row['source_video_sha256'] is not None,
            "missing_provenance", "decoded index needs source video hash")
    capture = row['capture_mono_ns']
    if row['alignment_status'] == 'exact':
        require(_integer(capture, 1) and row['decoded_frame_index'] is not None
                and row['capture_time_provenance'] != 'unknown' and row['source_log_sha256'] is not None,
                "capture_time_inconsistency", "exact capture needs video/log provenance")
    else:
        require(capture is None and row['capture_time_provenance'] == 'unknown', "capture_time_inconsistency")
    value = row['value_kph']
    if row['value_state'] == 'read':
        require(type(value) is int and MIN_VALUE_KPH <= value <= MAX_VALUE_KPH, "illegal_speed")
        require(row['sign_family'] in {'maximum_speed', 'cancellation', 'zone', 'variable_display', 'road_marking'}, "value_family_mismatch")
    else:
        require(value is None, "value_state_inconsistency")
    box = row['bbox_xyxy']
    if box is None:
        require(row['sign_family'] == 'not_a_sign' and row['value_state'] == 'not_applicable', "invalid_bbox", "only reviewed negative frames may omit a box")
    else:
        require(isinstance(box, list) and len(box) == 4
                and all(type(x) in (int, float) and math.isfinite(x) for x in box), "invalid_bbox")
        x1, y1, x2, y2 = box
        require(0 <= x1 < x2 <= row['width'] and 0 <= y1 < y2 <= row['height'], "invalid_bbox")
    score = row['proposal_score']
    require(score is None or (type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1), "invalid_score")
    require(isinstance(row['image_path'], str) and bool(row['image_path']), "missing_image")
    try:
        path = root / normalize_relpath(root, Path(row['image_path']))
    except ValueError as exc:
        raise DatasetError('unsafe_image_path', str(exc)) from exc
    require(path.is_file(), 'missing_image', row['image_path'])
    require(sha256_of(path) == row['image_sha256'], 'hash_mismatch', row['image_path'])
    try:
        dimensions = png_dimensions(path, verify=True)
    except MediaToolError as exc:
        raise DatasetError('invalid_image', str(exc)) from exc
    require(dimensions == (row['width'], row['height']), 'image_dimensions_mismatch', row['image_path'])


def validate_dataset(rows: list[dict], root: Path, *, dataset_kind: str = 'gold') -> dict:
    # [model-review] - START
    require(dataset_kind in DATASET_KINDS, 'unknown_dataset_kind', str(dataset_kind))
    # [model-review] - END
    errors = []
    seen, boxes, frames = set(), set(), {}
    negative_frames, object_frames = set(), set()
    for line, row in enumerate(rows, 1):
        try:
            validate_annotation(row, root, dataset_kind=dataset_kind)
            require(row['annotation_id'] not in seen, 'duplicate_identity', row['annotation_id'])
            seen.add(row['annotation_id'])
            identity = frame_identity(row)
            evidence = tuple(row[k] for k in ('image_sha256', 'decoded_frame_index', 'capture_mono_ns', 'source_video_sha256', 'source_log_sha256'))
            require(identity not in frames or frames[identity] == evidence, 'capture_time_inconsistency', 'inconsistent frame evidence')
            frames[identity] = evidence
            if row['review_state'] != 'rejected':
                box_identity = identity + (tuple(row['bbox_xyxy']) if row['bbox_xyxy'] else None,)
                require(box_identity not in boxes, 'duplicate_identity', 'same frame and box')
                boxes.add(box_identity)
                if row['bbox_xyxy'] is None:
                    negative_frames.add(identity)
                else:
                    object_frames.add(identity)
                require(not (identity in negative_frames and identity in object_frames),
                        'contradictory_frame_labels', 'whole-frame negative coexists with retained object labels')
        except (ValueError, OSError) as exc:
            errors.append({'line': line, 'reason': getattr(exc, 'reason', 'invalid_schema'), 'detail': str(exc)})
    clocks = {}
    for identity, evidence in sorted(frames.items()):
        capture = evidence[2]
        if capture is None:
            continue
        route, segment, stream, output = identity
        key = route, stream
        if key in clocks and capture <= clocks[key]:
            errors.append({'reason': 'capture_time_inconsistency', 'detail': f'nonmonotonic capture for {route}/{stream}'})
        clocks[key] = capture
    if not rows:
        errors.append({'reason': 'empty_dataset', 'detail': 'no reviewed annotations'})
    # [model-review] - START
    return {'schema_version': SCHEMA_VERSION, 'dataset_kind': dataset_kind, 'valid': not errors,
    # [model-review] - END
            'annotation_count': len(rows),
            'frame_count': len(frames), 'timing_eligible_count': sum(r.get('alignment_status') == 'exact' and r.get('review_state') in ('accepted','corrected') for r in rows),
            'errors': errors, 'dataset_sha256': hashlib.sha256(jsonl_bytes(rows)).hexdigest(),
            'warnings': ['site/encounter groups are incomplete; repeated physical sites across routes need human grouping']
                        if any(r.get('site_group') is None for r in rows) else []}


def _canonical(frame: dict, detection: dict | None, original: dict | None, source_hash: str,
               *, rejected: bool = False) -> dict:
    negative = detection is None
    d = detection or {}
    family = d.get('family', 'not_a_sign')
    require(isinstance(family, str), 'unknown_enum', 'sign_family')
    family = {'road_marking_candidate': 'road_marking'}.get(family, family)
    value = d.get('value_kph')
    value_state = 'read' if value is not None else 'not_applicable' if negative or family == 'not_a_sign' else 'unreadable' if family == 'unreadable' else 'unknown'
    label_keys = ('bbox_xyxy', 'family', 'value_kph', 'applicability', 'supplementary_panel', 'visibility', 'occlusion', 'temporary', 'variable_display')
    unchanged = original is not None and all(d.get(k) == original.get(k) for k in label_keys)
    object_id = 'negative-frame' if negative else d.get('detection_id')
    require(isinstance(object_id, str) and bool(object_id), 'missing_provenance', 'detection_id')
    frame_key = {k: frame.get(k) for k in ('route_id','segment_index','camera_stream','output_index','decoded_frame_index')}
    identity = [*frame_identity(frame_key), object_id]
    # [model-review] - START
    model = frame.get('label_basis') == 'model_review'
    provenance = {'label_basis': 'model_review' if model else 'human_review', 'source_review_sha256': source_hash,
                  'original_proposal': original, 'source_model': frame.get('model'), 'run_id': frame.get('run_id')}
    if model:
        provenance.update({k: frame[k] for k in ('review_model', 'review_model_version', 'review_model_sha256', 'confidence')})
    # [model-review] - END
    return {
        'schema_version': MODEL_SCHEMA_VERSION if model else SCHEMA_VERSION,
        'annotation_id': hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
        **{k: frame.get(k) for k in ('route_id','segment_index','output_index','decoded_frame_index','capture_mono_ns','camera_stream',
                                     'image_path','width','height','reviewer','review_timestamp','source_video_sha256','source_log_sha256','site_group','encounter_id')},
        'image_sha256': frame.get('frame_sha256'), 'bbox_xyxy': d.get('bbox_xyxy'), 'sign_family': family,
        'value_state': value_state, 'value_kph': value,
        'applicability': d.get('applicability', 'unresolved'), 'supplementary_panel': d.get('supplementary_panel', 'unknown'),
        'visibility': d.get('visibility', 'unknown'), 'occlusion': d.get('occlusion', 'unknown'),
        'temporary': d.get('temporary'), 'variable_display': d.get('variable_display'),
        'proposal_source': 'preannotation:' + str(frame.get('run_id')) if original else 'human',
        'proposal_score': original.get('score') if original else None,
        'review_state': 'rejected' if rejected else 'accepted' if negative or unchanged else 'corrected',
        'notes': d.get('notes', frame.get('notes', '')),
        'alignment_status': frame.get('alignment_status', 'unresolved'),
        'capture_time_provenance': frame.get('capture_time_provenance', 'unknown'),
        'provenance': provenance,
    }


def import_reviews(inputs: list[Path], root: Path, output: Path, *, dataset_kind: str = 'gold') -> dict:
    # [model-review] - START
    require(dataset_kind in DATASET_KINDS, 'unknown_dataset_kind', str(dataset_kind))
    # [model-review] - END
    imported = []
    frame_ids = set()
    sources = []
    for path in inputs:
        content = path.read_bytes()
        source_hash = hashlib.sha256(content).hexdigest()
        sources.append((source_hash, content))
        for frame in load_jsonl(path):
            if 'annotation_id' in frame:
                imported.append(frame)
                continue
            # [model-review] - START
            require(frame.get('kind') == 'reviewed_frame' and frame.get('review_state') in ('reviewed', 'resolved')
                    and frame.get('label_basis') in ('human_review', 'model_review'), 'unreviewed_proposal', path.name)
            if frame.get('label_basis') == 'model_review':
                require(dataset_kind == 'training-candidate', 'model_review_not_gold', path.name)
            # [model-review] - END
            require(isinstance(frame.get('route_id'), str) and isinstance(frame.get('camera_stream'), str)
                    and _integer(frame.get('segment_index')) and _integer(frame.get('output_index')),
                    'invalid_identity', 'reviewed frame')
            identity = tuple(frame.get(k) for k in ('route_id','segment_index','camera_stream','output_index'))
            require(identity not in frame_ids, 'duplicate_identity', 'reviewed frame')
            frame_ids.add(identity)
            detections, proposals = frame.get('detections'), frame.get('source_proposals')
            require(isinstance(detections, list) and isinstance(proposals, list)
                    and all(isinstance(d, dict) for d in detections + proposals), 'invalid_schema', 'detections/source_proposals')
            require(all(isinstance(d.get('detection_id'), str) and d['detection_id'] for d in proposals),
                    'invalid_identity', 'proposal id')
            by_id = {d.get('detection_id'): d for d in proposals}
            require(len(by_id) == len(proposals) and None not in by_id, 'duplicate_identity', 'proposal id')
            ids = [d.get('detection_id') for d in detections]
            require(all(isinstance(i,str) and i for i in ids) and len(set(ids)) == len(ids), 'duplicate_identity', 'detection id')
            imported.extend(_canonical(frame, d, by_id.get(d['detection_id']), source_hash) for d in detections)
            if not detections:
                imported.append(_canonical(frame, None, None, source_hash))
            imported.extend(_canonical(frame, d, d, source_hash, rejected=True) for k,d in by_id.items() if k not in ids)
    require(bool(imported), 'empty_dataset', 'no annotations to import')
    existing = load_jsonl(output) if output.exists() else []
    combined = sorted(existing + imported, key=lambda r: str(r.get('annotation_id', '')))
    # [model-review] - START
    report = validate_dataset(combined, root, dataset_kind=dataset_kind)
    # [model-review] - END
    if not report['valid']:
        raise DatasetError(report['errors'][0]['reason'], json.dumps(report, sort_keys=True))
    # Preserve exact source reviews; failed validation never publishes dataset truth.
    for digest, payload in sources:
        archived = root / 'annotations/imports' / (digest + '.jsonl')
        if archived.exists():
            require(archived.read_bytes() == payload, 'source_archive_mismatch')
        else:
            atomic_write(archived, payload)
    atomic_write(output, jsonl_bytes(combined))
    return {**report, 'imported_count': len(imported), 'output': str(output)}
# [reviewed-dataset] - END
