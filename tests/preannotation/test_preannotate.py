# [preannotation] - START
import json
import math
from pathlib import Path

import pytest

from nnslr_tools.preannotate import (
    MODEL_SHA256, PreannotationError, discover_frames, resolve_data_root,
    semantic_label, suppress_overlaps, unletterbox, run_preannotation,
)

ROUTE = '00000089--0ac1c0fdec'


def frames(root, segment=0, count=1):
    directory = root / 'derived/frames' / ROUTE / str(segment)
    directory.mkdir(parents=True)
    rows = []
    for index in range(count):
        image = directory / f'frame_{index:08}.png'
        image.write_bytes(b'fake image')
        rows.append({'image_path': str(image), 'route_id': ROUTE,
                     'segment_index': segment, 'output_index': index,
                     'requested_sample_fps': 1, 'start_s': None})
    (directory / 'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return directory


def test_root_and_cli_have_defaults_without_environment(monkeypatch):
    from nnslr_tools.cli import build_parser
    monkeypatch.delenv('NNSLR_DATA_ROOT', raising=False)
    assert resolve_data_root() == Path('/srv/nnslr-data')
    args = build_parser().parse_args(['preannotate', '--route', ROUTE])
    assert args.route == ROUTE and args.data_root is None
    assert not getattr(args, 'output', None)
    review = build_parser().parse_args(['review', '--route', ROUTE])
    assert review.host == '127.0.0.1' and review.port == 8765


def test_native_coordinates_use_real_resized_dimensions_and_crop_offset():
    # 1344x760 -> 640x362, top padding139; a centered 100x50 model box.
    box = unletterbox([320, 320, 100, 50], (1344, 760), (640, 362), (0, 139), (0, 0))
    assert box == pytest.approx([567, 327.513812, 777, 432.486188])
    assert unletterbox([20, 20, 60, 60], (100,100), (100,100), (0,0), (200,50)) == [200,50,250,100]
    assert unletterbox([1,1,2,2], (100,100), (100,100), (0,10), (0,0)) is None


def test_speed_mapping_is_exact_not_digits_from_arbitrary_signs():
    assert semantic_label('Speed limit 50km/h') == ('maximum_speed', 50)
    assert semantic_label('End of 50km/h speed limit') == ('cancellation', None)
    assert semantic_label('End of all prohibition') == ('cancellation', None)
    assert semantic_label('No Two or Three-wheeled Vehicles') == ('other_sign', None)
    assert semantic_label('Speed limit 0km/h') == ('other_sign', None)


def test_nms_suppresses_same_object_and_is_deterministic():
    a = {'bbox_xyxy': [10,20,30,40], 'score': .8, 'class_id': 2}
    b = dict(a, score=.9)
    c = dict(a, class_id=12, score=.85)
    assert suppress_overlaps([a,b,c], .5) == [b,c]
    with pytest.raises(PreannotationError):
        suppress_overlaps([dict(a, score=math.nan)], .5)


def test_frames_sorted_by_numeric_segment_and_keep_sample_identity(tmp_path):
    frames(tmp_path, 10); frames(tmp_path, 2, 2)
    rows = discover_frames(tmp_path, ROUTE)
    assert [(r['segment_index'], r['output_index']) for r in rows] == [(2,0),(2,1),(10,0)]
    assert all(r['capture_mono_ns'] is None and r['decoded_frame_index'] is None for r in rows)
    assert all(r['alignment_status'] == 'unresolved' for r in rows)


def test_manifest_cannot_reference_another_route_or_escape_root(tmp_path):
    directory = frames(tmp_path)
    manifest = directory / 'frames.jsonl'
    row = json.loads(manifest.read_text())
    row['image_path'] = '/etc/passwd'
    manifest.write_text(json.dumps(row)+'\n')
    with pytest.raises(PreannotationError, match='outside'):
        discover_frames(tmp_path, ROUTE)


def test_missing_frames_does_not_download_model(tmp_path, monkeypatch):
    monkeypatch.setattr('nnslr_tools.preannotate.ensure_model', lambda *a: pytest.fail('unexpected download'))
    with pytest.raises(PreannotationError, match='extract-frames'):
        run_preannotation(ROUTE, data_root=tmp_path)


def test_corrupt_cached_model_is_rejected(tmp_path):
    from nnslr_tools.preannotate import ensure_model, model_path
    path = model_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b'wrong model')
    with pytest.raises(PreannotationError, match='checksum'):
        ensure_model(tmp_path)


def test_review_handler_blocks_other_files_and_symlinks(tmp_path):
    from nnslr_tools.review import public_file
    (tmp_path/'index.html').write_text('report')
    (tmp_path/'images').mkdir()
    (tmp_path/'images'/'escape.jpg').symlink_to('/etc/passwd')
    assert public_file(tmp_path, '/') == tmp_path/'index.html'
    for path in ['/../secret','/run.json','/images/escape.jpg','/images/../index.html','/missing']:
        assert public_file(tmp_path, path) is None
# [preannotation] - END

# [preannotation] - START
def test_tile_coverage_includes_edges_without_duplicate_full_frame():
    from nnslr_tools._preannotate_worker import tile_boxes
    assert tile_boxes(320,240)==[(0,0,320,240)]
    tiles=tile_boxes(1344,760)
    assert len(tiles)==7 and len(set(tiles))==7
    assert (704,120,1344,760) in tiles


def test_download_validates_before_promoting(tmp_path,monkeypatch):
    import hashlib,io
    from nnslr_tools.preannotate import ensure_model,model_path
    payload=b'test ONNX bytes'
    monkeypatch.setattr('nnslr_tools.preannotate.MODEL_SHA256',hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr('nnslr_tools.preannotate._open_download',lambda:io.BytesIO(payload))
    path=ensure_model(tmp_path)
    assert path.read_bytes()==payload
    assert not list(path.parent.glob('.download-*'))
    monkeypatch.setattr('nnslr_tools.preannotate._open_download',lambda:pytest.fail('cached model redownloaded'))
    assert ensure_model(tmp_path)==path


def test_incomplete_worker_keeps_previous_report(tmp_path,monkeypatch):
    import subprocess
    from nnslr_tools.preannotate import run_preannotation
    frames(tmp_path)
    base=tmp_path/'derived/preannotations'/ROUTE;base.mkdir(parents=True)
    previous='{"run_id":"previous"}'
    (base/'latest.json').write_text(previous)
    monkeypatch.setattr('nnslr_tools.preannotate.ensure_model',lambda root:root/'model.onnx')
    monkeypatch.setattr('nnslr_tools.preannotate.ensure_road_model',lambda root:root/'road.onnx')
    def child(args,**kwargs):
        return subprocess.CompletedProcess(args,0 if '-c' in args else 1,'','')
    monkeypatch.setattr('nnslr_tools.preannotate.subprocess.run',child)
    with pytest.raises(PreannotationError,match='previous results preserved'):
        run_preannotation(ROUTE,data_root=tmp_path)
    assert (base/'latest.json').read_text()==previous


def test_report_escapes_embedded_json_and_is_self_contained(tmp_path):
    from nnslr_tools.review import write_review
    write_review(tmp_path,[{'model_label':'</script><script>bad()</script>'}],'run')
    text=(tmp_path/'index.html').read_text()
    assert '</script><script>bad()' not in text
    assert '__NNSLR_DATA__' not in text
    assert 'https://' not in text
# [preannotation] - END

# [road-text] - START
def test_road_text_proposal_maps_stretched_ocr_back_to_native_pixels():
    from nnslr_tools.road_text import road_text_proposals
    result=[[[[654,316],[726,316],[726,348],[654,348]],'30',.983]]
    proposal=road_text_proposals(result, width=1344, height=760, road_mask=[[True]*1344 for _ in range(760)])[0]
    assert proposal['bbox_xyxy']==pytest.approx([654,485.333333,726,496])
    assert proposal['value_kph']==30
    assert proposal['family']=='road_marking_candidate'
    assert proposal['evidence_source']=='ocr_lower_half_stretch3'
    assert proposal['applicability']=='unresolved'
    assert proposal['review_state']=='pending'


def test_ocr_does_not_promote_plates_text_or_low_confidence_to_speed():
    from nnslr_tools.road_text import road_text_proposals
    box=[[100,100],[200,100],[200,140],[100,140]]
    for text,score in [('AG-108816',.99),('ZONE30',.99),('0',.99),('1234',.99),('30',.5),('O',.99),(' 30x',.99)]:
        assert road_text_proposals([[box,text,score]],width=1344,height=760,road_mask=[[True]*1344 for _ in range(760)])==[]
    assert road_text_proposals([[box,'50',.95]],width=1344,height=760,road_mask=[[True]*1344 for _ in range(760)])[0]['value_kph']==50
# [road-text] - END

# [road-text] - START
def test_numeric_text_on_wall_advertisement_or_unknown_surface_is_rejected():
    from nnslr_tools.road_text import road_text_proposals
    box=[[40,30],[80,30],[80,60],[40,60]]
    reading=[[box,'30',.999]]
    for surface in (None, [[False]*200 for _ in range(100)]):
        assert road_text_proposals(reading,width=200,height=100,road_mask=surface)==[]
    # A center pixel on road is insufficient: the whole box and its context matter.
    mask=[[False]*200 for _ in range(100)]
    mask[65][60]=True
    assert road_text_proposals(reading,width=200,height=100,road_mask=mask)==[]
    mask=[[True]*200 for _ in range(100)]
    assert len(road_text_proposals(reading,width=200,height=100,road_mask=mask))==1


def test_road_text_rejects_invalid_boxes_and_uses_context_outside_text():
    from nnslr_tools.road_text import road_text_proposals
    mask=[[False]*200 for _ in range(100)]
    for y in range(60,70):
        for x in range(40,80):
            mask[y][x]=True
    box=[[40,30],[80,30],[80,60],[40,60]]
    assert road_text_proposals([[box,'30',.99]],width=200,height=100,road_mask=mask)==[]
    mask=[[True]*200 for _ in range(100)]
    for bad in ([[float('nan'),30],[80,30],[80,60],[40,60]],
                [[-1,30],[80,30],[80,60],[40,60]],
                [[40,30],[40,30],[40,30],[40,30]]):
        assert road_text_proposals([[bad,'30',.99]],width=200,height=100,road_mask=mask)==[]
# [road-text] - END
