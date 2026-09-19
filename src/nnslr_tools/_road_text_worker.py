# [road-text] - START
"""Optional CPU models; imported only from the inference subprocess."""
from importlib.metadata import version
from pathlib import Path

from nnslr_tools.manifest import sha256_of
from nnslr_tools.road_text import OCR_ASSETS, ROAD_SHA256, road_text_proposals


class RoadTextDetector:
    def __init__(self, model_path):
        import onnxruntime as ort
        import rapidocr
        if sha256_of(Path(model_path)) != ROAD_SHA256:
            raise ValueError('road model checksum mismatch')
        if version('rapidocr') != '3.4.0':
            raise ValueError('road OCR requires rapidocr==3.4.0')
        assets = Path(rapidocr.__file__).parent/'models'
        for name, checksum in OCR_ASSETS.items():
            if sha256_of(assets/name) != checksum:
                raise ValueError(f'OCR model checksum mismatch: {name}')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.surface = ort.InferenceSession(str(model_path), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.ocr = rapidocr.RapidOCR(params={
            'Global.use_cls': False, 'Global.log_level': 'error',
            'EngineConfig.onnxruntime.intra_op_num_threads': 2,
            'EngineConfig.onnxruntime.inter_op_num_threads': 1,
            'EngineConfig.onnxruntime.use_cuda': False,
            'EngineConfig.onnxruntime.use_dml': False,
            'EngineConfig.onnxruntime.use_cann': False,
            'Det.model_path': str(assets/'ch_PP-OCRv4_det_infer.onnx'),
            'Rec.model_path': str(assets/'ch_PP-OCRv4_rec_infer.onnx'),
            'Cls.model_path': str(assets/'ch_ppocr_mobile_v2.0_cls_infer.onnx'),
        })

    def road_mask(self, image):
        import numpy as np
        from PIL import Image
        # Pinned export's preprocessor_config.json: RGB, bilinear 512x512,
        # 1/255 rescale and ImageNet normalization; Cityscapes road is class 0.
        tensor = np.asarray(image.resize((512,512),Image.Resampling.BILINEAR),dtype=np.float32)/255
        tensor = (tensor-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        logits = self.surface.run(None,{'pixel_values':tensor.transpose(2,0,1)[None]})[0]
        if logits.shape != (1,19,128,128) or not np.isfinite(logits).all():
            raise ValueError('invalid road segmentation output')
        # Interpolate logits before softmax/argmax, preserving native coordinates.
        logits = np.stack([np.asarray(Image.fromarray(z).resize(image.size,Image.Resampling.BILINEAR))
                           for z in logits[0]])
        exp = np.exp(logits-logits.max(0))
        return (logits.argmax(0)==0) & (exp[0]/exp.sum(0)>=.9)

    def detect(self, image):
        import numpy as np
        from PIL import Image
        width,height = image.size
        mask = self.road_mask(image)
        crop = image.crop((0,height//2,width,height))
        stretched = crop.resize((width,crop.height*3),Image.Resampling.BICUBIC)
        result = self.ocr(np.asarray(stretched)[:,:,::-1].copy())
        readings = [] if result.boxes is None else zip(result.boxes,result.txts,result.scores)
        return road_text_proposals(readings,width=width,height=height,road_mask=mask)
# [road-text] - END
