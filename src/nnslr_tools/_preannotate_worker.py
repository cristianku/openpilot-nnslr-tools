# [preannotation] - START
"""Optional CPU ONNX worker, invoked only in its own interpreter."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from nnslr_tools.preannotate import MODEL_SHA256, semantic_label, suppress_overlaps, unletterbox
from nnslr_tools.manifest import sha256_of


def tile_boxes(width, height, size=640, overlap=128):
    def starts(length):
        if length <= size:
            return [0]
        return sorted(set([*range(0,length-size+1,size-overlap),length-size]))
    full=(0,0,width,height)
    return [full]+[b for y in starts(height) for x in starts(width)
                  if (b:=(x,y,min(width,x+size),min(height,y+size))) != full]


def run(request):
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    if sha256_of(Path(request['model_path'])) != MODEL_SHA256:
        raise ValueError('model checksum mismatch in worker')
    options=ort.SessionOptions(); options.intra_op_num_threads=2
    session=ort.InferenceSession(request['model_path'],sess_options=options,providers=['CPUExecutionProvider'])
    if session.get_inputs()[0].shape != [1,3,640,640] or session.get_outputs()[0].shape != [1,86,8400]:
        raise ValueError('unexpected model input/output contract')
    names=ast.literal_eval(session.get_modelmeta().custom_metadata_map['names'])
    if set(names)!=set(range(82)) or not all(isinstance(v,str) for v in names.values()):
        raise ValueError('invalid class metadata')
    from nnslr_tools._road_text_worker import RoadTextDetector
    road_text=RoadTextDetector(request['road_model_path'])
    output=Path(request['output']); (output/'images').mkdir()
    root=Path(request['data_root'])
    with (output/'preannotations.jsonl').open('w') as manifest:
        for ordinal,frame in enumerate(request['frames']):
            source=root/frame['image_path']
            if sha256_of(source)!=frame['frame_sha256']:
                raise ValueError(f'frame changed during inference: {source}')
            with Image.open(source) as raw:
                image=raw.convert('RGB')
            width,height=image.size; detections=[]
            for tile in tile_boxes(width,height):
                crop=image.crop(tile); w,h=crop.size
                scale=min(640/w,640/h); rw,rh=round(w*scale),round(h*scale)
                px,py=(640-rw)//2,(640-rh)//2
                padded=Image.new('RGB',(640,640),(114,114,114))
                padded.paste(crop.resize((rw,rh),Image.Resampling.BILINEAR),(px,py))
                tensor=np.asarray(padded,dtype=np.float32).transpose(2,0,1)[None]/255.0
                predictions=session.run(None,{'images':tensor})[0][0].T
                if predictions.shape != (8400,86) or not np.isfinite(predictions).all():
                    raise ValueError('invalid model output')
                class_ids=predictions[:,4:].argmax(1)
                scores=predictions[np.arange(len(predictions)),class_ids+4]
                for i in np.flatnonzero(scores>=request['confidence']):
                    box=unletterbox(predictions[i,:4],(w,h),(rw,rh),(px,py),tile[:2])
                    if box is None:
                        continue
                    class_id=int(class_ids[i]); family,value=semantic_label(names[class_id])
                    detections.append({'bbox_xyxy':box,'class_id':class_id,'model_label':names[class_id],
                                       'score':float(scores[i]),'family':family,'value_kph':value,
                                       'value_state':'proposed' if value is not None else 'unknown',
                                       'review_state':'pending','applicability':'unresolved',
                                       'supplementary_panel':'unknown'})
            detections=suppress_overlaps(detections)
            detections.extend(road_text.detect(image))
            for i,item in enumerate(detections):
                item['detection_id']=f"{frame['frame_key']}/{i}"
            preview=f'images/{ordinal:08}.jpg'
            image.save(output/preview,quality=92)
            row={**frame,'schema_version':1,'kind':'preannotated_frame','run_id':request['run_id'],
                 'width':width,'height':height,'preview_path':preview,'detections':detections,
                 'review_state':'pending','ground_truth':False,'model':request['model'],
                 'road_model':request['road_model'],'ocr':request['ocr'],
                 'inference':{'provider':'CPUExecutionProvider','confidence':request['confidence'],
                              'nms_iou':.5,'tiles':len(tile_boxes(width,height))}}
            manifest.write(json.dumps(row,allow_nan=False)+'\n'); manifest.flush()
            print(f"Preannotating {ordinal+1}/{len(request['frames'])}: segment {frame['segment_index']}, frame {frame['output_index']}, {len(detections)} proposals",file=sys.stderr,flush=True)


if __name__=='__main__':
    try:
        run(json.loads(Path(sys.argv[1]).read_text()))
    except Exception as exc:
        print(f'preannotation worker error: {exc}',file=sys.stderr)
        sys.exit(1)
# [preannotation] - END
