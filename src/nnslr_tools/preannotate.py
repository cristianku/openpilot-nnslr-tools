# [preannotation] - START
"""Route-based pretrained proposals. Parent process and helpers are stdlib-only."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from nnslr_tools.manifest import parse_route_id, sha256_of
from nnslr_tools.sync import DEFAULT_DATA_ROOT
from nnslr_tools.road_text import ROAD_MODEL_INFO, OCR_INFO

MODEL_ID = 'star092304/traffic-sign-detection-vietnam-yolo'
MODEL_REVISION = '58719e5c03bf5fa2817b59f9c7c3c385d3e92c9d'
MODEL_SHA256 = '9c171487bc80e00ba6a8dadaaec0914b0760b92511bc58f13328ef7f7f7a8063'
MODEL_URL = f'https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/best.onnx'
MODEL_INFO = {
    'repository': MODEL_ID, 'revision': MODEL_REVISION, 'sha256': MODEL_SHA256,
    'url': MODEL_URL, 'model_card_license': 'apache-2.0',
    'embedded_model_license': 'AGPL-3.0', 'training_domain': 'Vietnam traffic signs',
    'swiss_validation': 'not_established', 'input_size': [640, 640],
}


class PreannotationError(ValueError):
    pass


def resolve_data_root(explicit: Path | None = None) -> Path:
    return Path(explicit or os.environ.get('NNSLR_DATA_ROOT') or DEFAULT_DATA_ROOT).expanduser().resolve()


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise PreannotationError(f'path outside data root: {path}')
    return resolved


def discover_frames(root: Path, route: str) -> list[dict]:
    parse_route_id(route)
    directory = _inside(root, root / 'derived/frames' / route)
    manifests = sorted(directory.glob('*/frames.jsonl'), key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else -1)
    if not manifests:
        raise PreannotationError(f'no frame manifests; run: nnslr extract-frames --route {route}')
    rows = []
    seen = set()
    for manifest in manifests:
        _inside(directory, manifest)
        if not manifest.parent.name.isdigit():
            raise PreannotationError(f'non-numeric segment: {manifest.parent.name}')
        segment = int(manifest.parent.name)
        for line in manifest.read_text().splitlines():
            raw = json.loads(line)
            if not isinstance(raw, dict) or raw.get('route_id') != route or raw.get('segment_index') != segment:
                raise PreannotationError(f'frame identity mismatch: {manifest}')
            index = raw.get('output_index')
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or (segment,index) in seen:
                raise PreannotationError(f'invalid/duplicate output_index: {manifest}')
            name = raw.get('image_path')
            if not isinstance(name, str) or not name:
                raise PreannotationError(f'missing image_path: {manifest}')
            path = Path(name)
            path = _inside(directory, path if path.is_absolute() else root / path)
            if not path.is_file():
                raise PreannotationError(f'frame not found: {path}')
            # [frame-provenance] - START
            image_hash = sha256_of(path)
            if raw.get('image_sha256') is not None and raw['image_sha256'] != image_hash:
                raise PreannotationError(f'frame hash mismatch since extraction: {path}')
            # [frame-provenance] - END
            seen.add((segment,index))
            rows.append({
                'frame_key': f'{segment}/{index}', 'route_id': route, 'segment_index': segment,
                'output_index': index, 'image_path': str(path.relative_to(root)),
                'frame_sha256': image_hash, 'camera_stream': raw.get('camera_stream', 'narrow_road'),
                'requested_sample_fps': raw.get('requested_sample_fps'), 'start_s': raw.get('start_s'),
                # [frame-provenance] - START
                # Preserve explicit source evidence; legacy output_index is never promoted.
                'decoded_frame_index': raw.get('decoded_frame_index'),
                'capture_mono_ns': raw.get('capture_mono_ns'),
                'alignment_status': raw.get('alignment_status', 'unresolved'),
                'alignment_reason': raw.get('alignment_reason', 'legacy_manifest'),
                'capture_time_provenance': raw.get('capture_time_provenance', 'unknown'),
                'media_time_s': raw.get('media_time_s'),
                'media_time_provenance': raw.get('media_time_provenance', 'unknown'),
                'source_video_sha256': raw.get('source_video_sha256'),
                'source_log_sha256': raw.get('source_log_sha256'),
                # [frame-provenance] - END
            })
    if not rows:
        raise PreannotationError('frame manifests contain no frames')
    return sorted(rows, key=lambda r: (r['segment_index'],r['output_index']))


def model_path(root: Path) -> Path:
    return _inside(root, root / 'models/preannotation' / MODEL_SHA256 / 'model.onnx')


def _open_download(url=MODEL_URL):
    request = urllib.request.Request(url, headers={'User-Agent': 'nnslr-tools'})
    try:
        return urllib.request.urlopen(request, timeout=60)
    except urllib.error.URLError as exc:
        # Some macOS Python distributions omit the OS CA bundle. Still verify
        # TLS using that bundle; never disable certificate verification.
        ca = Path('/etc/ssl/cert.pem')
        if isinstance(exc.reason, ssl.SSLCertVerificationError) and ca.is_file():
            return urllib.request.urlopen(request, context=ssl.create_default_context(cafile=str(ca)), timeout=60)
        raise


def ensure_model(root: Path) -> Path:
    return _ensure_asset(model_path(root), MODEL_SHA256, MODEL_ID, _open_download)


def ensure_road_model(root: Path) -> Path:
    info = ROAD_MODEL_INFO
    path = _inside(root, root / "models/preannotation" / info["sha256"] / "model.onnx")
    return _ensure_asset(path, info["sha256"], info["repository"], lambda: _open_download(info["url"]))


def _ensure_asset(path, checksum, repository, opener):
    if path.exists():
        if sha256_of(path) != checksum:
            raise PreannotationError(f'model checksum mismatch: {path}; remove only this cached model and retry')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading pretrained ONNX model: {repository}', file=sys.stderr, flush=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.download-', delete=False) as target:
            temporary = Path(target.name)
            with opener() as source:
                while chunk := source.read(1024*1024):
                    target.write(chunk)
        if sha256_of(temporary) != checksum:
            raise PreannotationError('downloaded model checksum mismatch')
        temporary.replace(path)
    except (OSError, urllib.error.URLError) as exc:
        raise PreannotationError(f'model download failed: {exc}') from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def semantic_label(label: str) -> tuple[str, int | None]:
    match = re.fullmatch(r'Speed limit ([1-9][0-9]*)km/h', label)
    if match:
        return 'maximum_speed', int(match[1])
    if label in {'End of all prohibition', 'End of 50km/h speed limit'}:
        return 'cancellation', None
    return 'other_sign', None


def unletterbox(xywh, original_size, resized_size, padding, offset=(0,0)):
    """Map network pixels to native pixels using actual rounded resize sizes."""
    cx, cy, w, h = map(float, xywh)
    if not all(math.isfinite(x) for x in (cx,cy,w,h)) or w <= 0 or h <= 0:
        return None
    ow,oh = original_size; rw,rh = resized_size; px,py = padding; ox,oy = offset
    x1=max(0,min(ow,(cx-w/2-px)*ow/rw)); x2=max(0,min(ow,(cx+w/2-px)*ow/rw))
    y1=max(0,min(oh,(cy-h/2-py)*oh/rh)); y2=max(0,min(oh,(cy+h/2-py)*oh/rh))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1+ox,y1+oy,x2+ox,y2+oy]


def _iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/area if area else 0


def suppress_overlaps(detections: list[dict], threshold: float = .5) -> list[dict]:
    for d in detections:
        b=d['bbox_xyxy']; score=d['score']
        if len(b)!=4 or not all(math.isfinite(v) for v in [*b,score]) or not 0<=score<=1 or b[2]<=b[0] or b[3]<=b[1]:
            raise PreannotationError('invalid model detection')
    ordered=sorted(detections,key=lambda d:(-d['score'],d['class_id'],d['bbox_xyxy']))
    keep=[]
    for item in ordered:
        if not any(item['class_id']==other['class_id'] and _iou(item['bbox_xyxy'],other['bbox_xyxy'])>threshold for other in keep):
            keep.append(item)
    return keep


def run_preannotation(route: str, *, data_root: Path | None = None, confidence: float = .25,
                      python_executable: str | None = None) -> dict:
    if not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise PreannotationError('confidence must be in (0, 1]')
    root=resolve_data_root(data_root)
    frames=discover_frames(root,route)
    python=python_executable or sys.executable
    check=subprocess.run([python,'-c','import numpy, PIL, onnxruntime, rapidocr'],capture_output=True,text=True)
    if check.returncode:
        raise PreannotationError('preannotation dependencies missing; in the nnslr environment run: python -m pip install ".[preannotate]"')
    model=ensure_model(root)
    road_model=ensure_road_model(root)
    base=_inside(root,root/'derived/preannotations'/route)
    base.mkdir(parents=True,exist_ok=True)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    pending=base/('.pending-'+run_id); pending.mkdir()
    request={'schema_version':1,'run_id':run_id,'route_id':route,'data_root':str(root),'model_path':str(model),
             'model':MODEL_INFO,'road_model_path':str(road_model),'road_model':ROAD_MODEL_INFO,
             'ocr':OCR_INFO,'confidence':confidence,'frames':frames,'output':str(pending)}
    request_path=pending/'request.json'; request_path.write_text(json.dumps(request,allow_nan=False))
    proc=subprocess.run([python,'-m','nnslr_tools._preannotate_worker',str(request_path)],capture_output=False)
    if proc.returncode:
        raise PreannotationError(f'preannotation failed; previous results preserved; diagnostic request: {request_path}')
    report=pending/'preannotations.jsonl'
    rows=[json.loads(line) for line in report.read_text().splitlines()]
    if len(rows)!=len(frames) or [r['frame_key'] for r in rows]!=[r['frame_key'] for r in frames]:
        raise PreannotationError('worker returned incomplete or reordered frames')
    from nnslr_tools.review import write_review
    write_review(pending, rows, run_id)
    summary={'schema_version':1,'run_id':run_id,'route_id':route,'model':MODEL_INFO,
             'road_model':ROAD_MODEL_INFO,'ocr':OCR_INFO,
             'confidence':confidence,'device':'cpu','frame_count':len(rows),
             'frames_with_proposals':sum(bool(r['detections']) for r in rows),
             'proposal_count':sum(len(r['detections']) for r in rows),
             'speed_proposal_count':sum(d['family']=='maximum_speed' for r in rows for d in r['detections']),
             'road_marking_proposal_count':sum(d['family']=='road_marking_candidate' for r in rows for d in r['detections']),
             'review_state':'pending','ground_truth':False,'capture_time_established':False}
    (pending/'run.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    destination=base/run_id; pending.rename(destination)
    pointer=base/('.latest-'+uuid.uuid4().hex+'.json')
    pointer.write_text(json.dumps({'run_id':run_id})); pointer.replace(base/'latest.json')
    return {**summary,'output':str(destination),'report':str(destination/'index.html'),
            'review_command':f'nnslr review --route {route}'}
# [preannotation] - END
