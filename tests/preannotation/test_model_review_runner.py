# [model-review] - START
import hashlib
from pathlib import Path
import pytest
from nnslr_tools.model_review import audit_frame, candidate_frame, finish_decision

META = {'model_id': 'test-model', 'revision': 'revision-1', 'model_sha256': 'a'*64}


def answer(detected=False, value=None, family='not_a_sign', confidence=.95, **extra):
    return dict(detected=detected, value_kph=value, sign_family=family, confidence=confidence,
                applicability='unresolved', multiple_visible=False, proposal_matches_review=False, **extra)


def test_high_confidence_conflict_stays_unresolved():
    first = answer()
    second = answer(True, 30, 'zone')
    assert finish_decision([], first, second) == 'STILL_AMBIGUOUS'
    assert finish_decision([], answer(confidence=.5), second) == 'RESOLVED_FALSE_NEGATIVE'


def test_multiple_objects_never_become_single_label():
    first = answer(True, 30, 'zone'); first['multiple_visible'] = True
    assert finish_decision([], first, answer()) == 'STILL_AMBIGUOUS'


def frame(root):
    image = root / 'frame.png'
    image.write_bytes(b'synthetic-image-content')
    return dict(image_path='frame.png', frame_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                route_id='00000001--abcdef1234', segment_index=0, output_index=3,
                decoded_frame_index=None, camera_stream='narrow_road', detections=[], model={'sha256':'b'*64},
                width=32, height=24, alignment_status='unresolved', capture_mono_ns=None,
                capture_time_provenance='unknown')


def reviewer(answers):
    def call(path, method, box):
        return {'review': answers.pop(0), 'method': method, 'raw_response': 'synthetic'}
    return call


def test_review_preserves_missing_timing_and_exports_model_negative(tmp_path):
    source=frame(tmp_path)
    row=audit_frame(source, tmp_path, META, reviewer([answer(), answer()]))
    assert row['result'] == 'OK' and row['review_state'] == 'reviewed'
    assert row['frame_index'] is None and row['output_index'] == 3
    assert len(row['passes']) == 2 and row['nnslr']['available'] is False
    exported = candidate_frame(source, row)
    assert exported['label_basis'] == 'model_review' and exported['reviewer'] == 'codex/test-model'
    assert exported['detections'] == [] and not exported['ground_truth']


def test_false_negative_without_box_never_exports(tmp_path):
    source=frame(tmp_path)
    row=audit_frame(source,tmp_path,META,reviewer([answer(True,30,'zone'),answer(True,30,'zone')]))
    assert row['result'] == 'FALSE_NEGATIVE'
    assert candidate_frame(source,row) is None


def test_missing_image_or_prediction_is_unavailable(tmp_path):
    source=frame(tmp_path);source.pop('detections')
    row=audit_frame(source,tmp_path,META,reviewer([]))
    assert row['result']=='UNAVAILABLE' and row['review_state']=='unresolved'
    source=frame(tmp_path); (tmp_path/'frame.png').unlink()
    assert audit_frame(source,tmp_path,META,reviewer([]))['result']=='UNAVAILABLE'


def test_hash_change_is_unavailable(tmp_path):
    source=frame(tmp_path); (tmp_path/'frame.png').write_bytes(b'changed')
    assert audit_frame(source,tmp_path,META,reviewer([]))['result']=='UNAVAILABLE'


def test_unverified_box_cannot_become_a_training_label(tmp_path):
    source=frame(tmp_path)
    source['detections']=[dict(detection_id='a',family='zone',value_kph=30,score=.95,bbox_xyxy=[1,2,10,12])]
    row=audit_frame(source,tmp_path,META,reviewer([answer(True,30,'zone'),answer(True,30,'zone')]))
    assert candidate_frame(source,row) is None
    verified=answer(True,30,'zone');verified['proposal_matches_review']=True
    row=audit_frame(source,tmp_path,META,reviewer([answer(True,30,'zone'),verified]))
    exported=candidate_frame(source,row)
    assert exported['detections'][0]['bbox_xyxy']==[1,2,10,12]
    assert exported['detections'][0]['family']=='zone'


def test_paths_cannot_escape_data_root(tmp_path):
    source=frame(tmp_path);source['image_path']='../outside.png'
    row=audit_frame(source,tmp_path,META,reviewer([]))
    assert row['result']=='UNAVAILABLE'
# [model-review] - END

# [model-review] - START
def test_model_bundle_hash_must_match_assets():
    from nnslr_tools.model_review import verify_local_model
    with pytest.raises(ValueError, match='model_bundle_hash_mismatch'):
        verify_local_model({'files': {'weights.gguf': 'b'*64}, 'model_sha256': 'a'*64}, {})
# [model-review] - END

# [model-review] - START
def test_malformed_review_preserves_raw_response(tmp_path):
    def invalid(path, method, box):
        return {'review': None, 'raw_response': 'unparseable model output', 'error': 'invalid_json'}
    row=audit_frame(frame(tmp_path),tmp_path,META,invalid)
    assert row['result']=='UNAVAILABLE'
    assert row['passes'][0]['raw_response']=='unparseable model output'
# [model-review] - END

# [model-review] - START
def test_resumable_run_retries_unavailable_and_reuses_completed(tmp_path, monkeypatch):
    import argparse
    import json
    from nnslr_tools import model_review as module
    source=frame(tmp_path)
    manifest=tmp_path/'preannotations.jsonl';manifest.write_text(json.dumps(source)+'\n')
    meta=tmp_path/'model.json';meta.write_text(json.dumps(META))
    server=tmp_path/'server.json';server.write_text(json.dumps({'port':18089}))
    output=tmp_path/'review';output.mkdir()
    args=argparse.Namespace(model_metadata=meta,server_metadata=server,preannotations=[manifest],limit=0)
    monkeypatch.setattr(module,'verify_local_model',lambda *_: None)
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k: 'test-commit\n')
    calls=[]
    def unavailable(*args):
        calls.append('unavailable');return {'review':None,'raw_response':'broken'}
    monkeypatch.setattr(module,'LocalVisualReviewer',lambda _: unavailable)
    with pytest.raises(RuntimeError,match='unavailable_frame'):
        module.run(args,tmp_path,output)
    assert json.loads((output/'run.json').read_text())['status']=='failed'
    def ambiguous(*args):
        calls.append('ambiguous');return {'review':answer(True,None,'unreadable',.5),'raw_response':'unreadable'}
    monkeypatch.setattr(module,'LocalVisualReviewer',lambda _: ambiguous)
    assert module.run(args,tmp_path,output)==0
    assert json.loads((output/'final/validation.json').read_text())['valid'] is False
    assert (output/'final/report/ambiguous/records.jsonl').is_file()
    assert module.run(args,tmp_path,output)==0
    assert calls==['unavailable','ambiguous','ambiguous']
    (tmp_path/'frame.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint_source_changed'):
        module.run(args,tmp_path,output)
# [model-review] - END
