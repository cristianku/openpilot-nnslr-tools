# [road-text] - START
"""Conservative OCR proposals gated by independently inferred road surface."""
from __future__ import annotations

import math
import re

ROAD_MODEL_ID = 'Xenova/segformer-b0-finetuned-cityscapes-1024-1024'
ROAD_REVISION = '64e537ca2bf6bf2afc13727ee6bb61cfca9e1048'
ROAD_SHA256 = '142427caa45a47d57d5bd23b62c3bfc852fa7df0c6c2291c5ef34f581fa406bf'
ROAD_MODEL_INFO = {
    'repository': ROAD_MODEL_ID, 'revision': ROAD_REVISION, 'sha256': ROAD_SHA256,
    'url': f'https://huggingface.co/{ROAD_MODEL_ID}/resolve/{ROAD_REVISION}/onnx/model.onnx',
    'license': 'NVIDIA SegFormer: non-commercial research/evaluation only',
    'input_size': [512, 512], 'road_class': 0, 'road_pixel_threshold': .9,
    'box_context_coverage_threshold': .95, 'swiss_validation': 'not_established',
}
OCR_ASSETS = {
    'ch_PP-OCRv4_det_infer.onnx': 'd2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9',
    'ch_PP-OCRv4_rec_infer.onnx': '48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b',
    'ch_ppocr_mobile_v2.0_cls_infer.onnx': 'e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c',
}
OCR_INFO = {'package': 'rapidocr', 'version': '3.4.0', 'model': 'PP-OCRv4',
            'license': 'Apache-2.0', 'assets_sha256': OCR_ASSETS,
            'crop': 'lower_half', 'vertical_stretch': 3, 'text_score_threshold': .9}


def road_text_proposals(results, *, width: int, height: int, road_mask=None) -> list[dict]:
    """Mask uses native pixels, True only for confident road. Missing mask rejects.

    OCR coordinates are relative to the lower half, stretched vertically 3x.
    Text and surrounding context must both lie inside the road prediction.
    This is evidence for human annotation, never an applicable speed limit.
    """
    if road_mask is None:
        return []
    if len(road_mask) != height or any(len(row) != width for row in road_mask):
        raise ValueError('road mask must match native image dimensions')
    proposals = []
    for quad, text, score in results:
        if not isinstance(text, str) or not re.fullmatch(r'[1-9][0-9]{0,2}', text):
            continue
        value = int(text)
        if value > 130 or not math.isfinite(score) or not .9 <= score <= 1:
            continue
        if len(quad) != 4 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in quad):
            continue
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1])/3 + height//2 for p in quad]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if not (0 <= x1 < x2 <= width and height//2 <= y1 < y2 <= height):
            continue
        # Include context around the number, not just a road-colored center pixel.
        padding = max(3, (y2-y1)*.5)
        left, top = math.floor(x1-padding), math.floor(y1-padding)
        right, bottom = math.ceil(x2+padding), math.ceil(y2+padding)
        if left < 0 or top < 0 or right > width or bottom > height:
            continue
        count = sum(bool(road_mask[y][x]) for y in range(top,bottom) for x in range(left,right))
        coverage = count / ((right-left)*(bottom-top))
        if coverage < .95:
            continue
        proposals.append({
            'bbox_xyxy': [x1,y1,x2,y2], 'class_id': None, 'model_label': text,
            'score': float(score), 'family': 'road_marking_candidate', 'value_kph': value,
            'value_state': 'proposed', 'review_state': 'pending', 'applicability': 'unresolved',
            'supplementary_panel': 'unknown', 'evidence_source': 'ocr_lower_half_stretch3',
            'surface_source': ROAD_MODEL_ID, 'road_context_coverage': coverage,
        })
    return proposals
# [road-text] - END
