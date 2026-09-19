# [reviewed-dataset] - START
import json
from test_annotations import reviewed, write, invoke


def make_dataset(root,capsys,count=8):
    rows=[reviewed(root,f'{i+1:08x}--abcdef1234',negative=i==0) for i in range(count)]
    source=write(root,rows)
    code,out=invoke(capsys,'import-annotations',str(source),'--data-root',str(root))
    assert code==0,out.err+out.out
    return root/'annotations/objects.jsonl'


def test_split_defaults_are_deterministic_and_route_disjoint(tmp_path,capsys):
    make_dataset(tmp_path,capsys)
    code,out=invoke(capsys,'build-splits','--data-root',str(tmp_path))
    assert code==0,out.err+out.out
    report=json.loads(out.out);path=tmp_path/report['split_path'];before=path.read_bytes()
    splits=json.loads(before)
    assert set(splits['assignments'].values())=={'train','validation','test','route-held-out','hard-negative'}
    assert len(splits['assignments'])==8
    assert splits['leakage_checked'] is True
    code,out=invoke(capsys,'build-splits','--data-root',str(tmp_path))
    assert code==0,out.err+out.out
    assert (tmp_path/json.loads(out.out)['split_path']).read_bytes()==before
    code,out=invoke(capsys,'validate-dataset','--data-root',str(tmp_path),'--splits',str(path))
    assert code==0,out.err+out.out


def test_too_few_independent_routes_fail_without_random_frame_split(tmp_path,capsys):
    make_dataset(tmp_path,capsys,2)
    code,out=invoke(capsys,'build-splits','--data-root',str(tmp_path))
    assert code!=0 and 'insufficient_independent_groups' in out.out+out.err
    assert not (tmp_path/'splits').exists()


def test_shared_sites_are_grouped_before_assignment(tmp_path,capsys):
    path=make_dataset(tmp_path,capsys)
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    for r in rows:
        if r['route_id'].startswith(('00000002','00000003')):r['site_group']='same-physical-site'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    code,out=invoke(capsys,'build-splits','--data-root',str(tmp_path))
    assert code==0,out.err+out.out
    result=json.loads((tmp_path/json.loads(out.out)['split_path']).read_text())
    assigned=[result['assignments'][r['annotation_id']] for r in rows if r['site_group']=='same-physical-site']
    assert len(set(assigned))==1
    # A manually modified split must be detected as leakage rather than accepted.
    targets=[r for r in rows if r['site_group']=='same-physical-site']
    result['assignments'][targets[0]['annotation_id']]='train'
    result['assignments'][targets[1]['annotation_id']]='test'
    changed=tmp_path/'leaked.json';changed.write_text(json.dumps(result))
    code,out=invoke(capsys,'validate-dataset','--data-root',str(tmp_path),'--splits',str(changed))
    assert code!=0 and 'split_leakage' in out.out
# [reviewed-dataset] - END

# [reviewed-dataset] - START
def test_invalid_split_values_are_reported_without_traceback(tmp_path,capsys):
    dataset=make_dataset(tmp_path,capsys)
    rows=[json.loads(s) for s in dataset.read_text().splitlines()]
    split=tmp_path/'invalid-splits.json'
    split.write_text(json.dumps({'schema_version':1,'assignments':{r['annotation_id']:[] for r in rows}}))
    code,out=invoke(capsys,'validate-dataset','--data-root',str(tmp_path),'--splits',str(split))
    assert code!=0 and 'invalid_split_membership' in out.out
# [reviewed-dataset] - END
