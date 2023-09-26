import os

import hydra
import logging
import numpy as np
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.checks import check_imgsz

from anprmodule.ocr import ocr_image
from .config_loader import get_config

logging.basicConfig()
logging.getLogger().setLevel(logging.FATAL)
reg = ''

class DetectionPredictor(BasePredictor):

    def preprocess(self, img):
        if isinstance(img, list):
            img = np.array(img)
            b, h, w, c = img.shape
            img = np.reshape(img, [b, c, h, w])

        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        self.seen += 1
        im0 = im0.copy()

        det = preds[idx]
        self.all_outputs.append(det)

        # Perform Image to Text analysis
        for *xyxy, conf, cls in reversed(det):
            self.text_ocr = ocr_image(im0, xyxy)

        return self.text_ocr

    def ret_reg(self):
        return self.text_ocr


def run(src, model):
    with hydra.initialize(version_base=None, config_path='.', job_name='anpr'):
        cfg = get_config()
        def _run(cfg):
            cfg.model = str(model)
            cfg.source = str(src)
            cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
            cfg.save = False  # Avoid saving detection outputs
            global reg
            predictor = DetectionPredictor(cfg)
            predictor()
            reg = predictor.ret_reg()
            return reg
        _run(cfg)
        return reg


if __name__ == "__main__":
    model = '../lib/best.pt'
    source = '../lib/EWP05W.jpg'
    r = run(src=source, model=model)
    print(r)
