# [model-review] - START
import json
from copy import deepcopy

import pytest

from nnslr_tools.model_review_audit import compare_review, second_pass_result, validate_review, write_report


def review(**changes):
    return dict(detected=True, value_kph=30, sign_family='maximum_speed', confidence=.95,
                applicability='own_road', **{}) | changes


def prediction(**changes):
    return dict(family='maximum_speed', value_kph=30, score=.8, bbox_xyxy=[1, 2, 20, 30]) | changes


@pytest.mark.parametrize('changes', [
    {'detected': 1}, {'value_kph': True}, {'confidence': True}, {'confidence': float('nan')},
    {'confidence': float('inf')}, {'confidence': 1.1}, {'value_kph': 0},
    {'detected': False}, {'sign_family': 'made_up'}, {'applicability': 'certain'},
])
def test_malformed_review_never_becomes_label(changes):
    with pytest.raises(ValueError):
        validate_review(review(**changes))


def test_missing_confidence_is_not_defaulted():
    row = review()
    del row['confidence']
    with pytest.raises(ValueError):
        validate_review(row)


@pytest.mark.parametrize('predictions,answer,result', [
    ([prediction()], review(), 'OK'),
    ([prediction()], review(value_kph=80), 'WRONG_VALUE'),
    ([prediction()], review(sign_family='zone'), 'WRONG_SIGN_FAMILY'),
    ([], review(), 'FALSE_NEGATIVE'),
    ([prediction()], review(detected=False, value_kph=None, sign_family='not_a_sign', applicability='unresolved'), 'FALSE_POSITIVE'),
    ([], review(detected=False, value_kph=None, sign_family='not_a_sign', applicability='unresolved'), 'OK'),
    ([prediction(), prediction()], review(), 'AMBIGUOUS'),
    ([prediction()], review(confidence=.4), 'LOW_CONFIDENCE'),
    ([prediction()], review(value_kph=None, sign_family='unreadable'), 'UNREADABLE'),
    ([prediction(applicability='own_road')], review(applicability='other_road'), 'APPLICABILITY_ERROR'),
    ([prediction()], review(applicability='other_road'), 'AMBIGUOUS'),
])
def test_comparison_taxonomy(predictions, answer, result):
    assert compare_review(predictions, answer) == result


def test_missing_prediction_is_not_a_negative():
    with pytest.raises(ValueError):
        compare_review([{}], review())
    with pytest.raises(ValueError):
        compare_review(None, review())


def test_second_pass_only_resolves_definite_evidence():
    assert second_pass_result('LOW_CONFIDENCE', 'WRONG_VALUE') == 'RESOLVED_WRONG_VALUE'
    assert second_pass_result('AMBIGUOUS', 'LOW_CONFIDENCE') == 'STILL_AMBIGUOUS'
    with pytest.raises(ValueError):
        second_pass_result('OK', 'WRONG_VALUE')


def audit(result='OK'):
    return dict(image='derived/<script>alert(1)</script>.png', image_sha256='a'*64,
                route_id='00000089--0ac1c0fdec', segment_index=2, frame_index=None, output_index=39,
                label_basis='model_review', reviewer='codex/test/model', review_model='test/model',
                review_model_version='a'*40, review_model_sha256='b'*64,
                review_timestamp='2026-09-19T13:00:00Z', result=result,
                pass1={'raw_response': '<script>bad()</script>'}, pass2=None)


def test_report_preserves_evidence_and_escapes_text(tmp_path):
    rows = [audit('LOW_CONFIDENCE'), audit('RESOLVED_WRONG_VALUE')]
    rows[1]['output_index'] = 40
    rows[1]['image'] = 'derived/40.png'
    destination = tmp_path/'report'
    result = write_report(rows, destination)
    assert result['frame_count'] == 2
    assert result['counts'] == {'LOW_CONFIDENCE': 1, 'RESOLVED_WRONG_VALUE': 1}
    assert [json.loads(x) for x in (destination/'audit.jsonl').read_text().splitlines()] == rows
    assert len((destination/'ambiguous/records.jsonl').read_text().splitlines()) == 1
    assert len((destination/'wrong_values/records.jsonl').read_text().splitlines()) == 1
    html = (destination/'report.html').read_text()
    assert '<script>' not in html and '&lt;script&gt;' in html
    with pytest.raises(FileExistsError):
        write_report(rows, destination)


def test_invalid_provenance_does_not_publish_partial_report(tmp_path):
    row = audit()
    del row['review_model_sha256']
    with pytest.raises(ValueError):
        write_report([row], tmp_path/'report')
    assert not (tmp_path/'report').exists()


def test_missing_model_is_explicitly_unavailable_not_invented(tmp_path):
    row = audit('UNAVAILABLE')
    row.update(review_model=None, review_model_version=None, review_model_sha256=None,
               reviewer=None, reason='review_model_missing')
    assert write_report([row], tmp_path/'report')['counts'] == {'UNAVAILABLE': 1}
    bad = deepcopy(row)
    bad['result'] = 'OK'
    with pytest.raises(ValueError):
        write_report([bad], tmp_path/'bad')
# [model-review] - END
