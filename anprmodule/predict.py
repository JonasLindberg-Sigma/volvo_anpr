import cv2
import hydra
import logging
import numpy as np
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.checks import check_imgsz


from omegaconf import OmegaConf

from anprmodule.ocr import ocr_image
#from config_loader import get_config

logging.basicConfig()
logging.getLogger().setLevel(logging.FATAL)
reg = ''
_DICT = {'task': 'detect', 'mode': 'train', 'model': None, 'data': None, 'epochs': 100, 'patience': 50, 'batch': 16, 'imgsz': 640, 'save': True, 'cache': False, 'device': None, 'workers': 8, 'project': None, 'name': None, 'exist_ok': False, 'pretrained': False, 'optimizer': 'SGD', 'verbose': False, 'seed': 0, 'deterministic': True, 'single_cls': False, 'image_weights': False, 'rect': False, 'cos_lr': False, 'close_mosaic': 10, 'resume': False, 'overlap_mask': True, 'mask_ratio': 4, 'dropout': 0.0, 'val': True, 'save_json': False, 'save_hybrid': False, 'conf': None, 'iou': 0.7, 'max_det': 300, 'half': False, 'dnn': False, 'plots': True, 'source': None, 'show': False, 'save_txt': False, 'save_conf': False, 'save_crop': False, 'hide_labels': False, 'hide_conf': False, 'vid_stride': 1, 'line_thickness': 3, 'visualize': False, 'augment': False, 'agnostic_nms': False, 'retina_masks': False, 'format': 'torchscript', 'keras': False, 'optimize': False, 'int8': False, 'dynamic': False, 'simplify': False, 'opset': 17, 'workspace': 4, 'nms': False, 'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'fl_gamma': 0.0, 'label_smoothing': 0.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0, 'hydra': {'output_subdir': None, 'run': {'dir': '${hydra:runtime.cwd}/${now:%Y-%m-%d_%H-%M-%S}'}}, 'v5loader': False}

def get_config():
    return OmegaConf.create(_DICT)
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
    source = ('truck2.jpg')
    r = run(src=source, model=model)
    print(r)
