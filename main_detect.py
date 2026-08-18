# YOLOv5 PyTorch 推理封装
import os
import sys
import logging
from pathlib import Path
import numpy as np
import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from base_detector import YOLOv5BaseDetector


class YOLOv5Detector(YOLOv5BaseDetector):
    """YOLOv5 PyTorch 目标检测器，封装模型加载、推理、绘制。

    Usage:
        detector = YOLOv5Detector('weights/yolov5s.pt', device='0')
        results = detector.detect(frame)
        for det in results:
            detector.draw(frame, det)
    """

    def __init__(self,
                 weights,
                 device='',
                 imgsz=(640, 640),
                 conf_thres=0.30,
                 iou_thres=0.45,
                 max_det=1000,
                 classes=None,
                 agnostic_nms=False,
                 augment=False,
                 visualize=False,
                 half=False,
                 dnn=False,
                 verbose=False,
                 names=None):
        """
        Args:
            weights: 权重文件路径 (.pt / .onnx / .engine 等)
            device: 设备，'0' / 'cpu' / '' (自动) / torch.device 对象
            imgsz: 输入图片尺寸 (h, w)
            conf_thres: 置信度阈值
            iou_thres: NMS IOU 阈值
            max_det: 单图最大检测数
            classes: 只保留指定类别 (None=全部)
            agnostic_nms: 类别无关 NMS
            augment: TTA 数据增强推理
            visualize: 特征图可视化
            half: FP16 半精度推理
            dnn: OpenCV DNN 做 ONNX 推理
            verbose: 是否显示 YOLOv5 默认输出
            names: 类别名字典 {id: name}，None 则从模型读取
        """
        # 兼容 torch.device 对象
        if isinstance(device, torch.device):
            device = '0' if device.type == 'cuda' else 'cpu'

        # 屏蔽 YOLOv5 默认输出
        if not verbose:
            logging.disable(logging.WARNING)

        try:
            self.device = select_device(device)
            self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn)
            imgsz = check_img_size(imgsz, s=self.model.stride)
            model_names = self.model.names

            # 半精度
            self.half = half and (self.model.pt or self.model.jit or self.model.onnx or self.model.engine) \
                        and self.device.type != 'cpu'
            if self.model.pt or self.model.jit:
                self.model.model.half() if self.half else self.model.model.float()

            # warmup 只执行一次
            self.model.warmup(imgsz=(1, 3, *imgsz))
        finally:
            if not verbose:
                logging.disable(logging.NOTSET)

        # 初始化基类（imgsz 已由 check_img_size 校验，names 优先用模型自带的）
        super().__init__(imgsz, conf_thres, iou_thres, max_det,
                         classes, agnostic_nms, names or model_names)

        self.augment = augment
        self.visualize = visualize

    def detect(self, img, **kwargs):
        """单张图片推理。

        Args:
            img: BGR 图片 (numpy array, cv2.imread 的结果)
            **kwargs: 临时覆盖推理参数，如 conf_thres=0.5, classes=[0]

        Returns:
            list[dict]: [{'class': str, 'conf': float, 'position': [l,t,w,h]}, ...]
        """
        p = {
            'conf_thres': self.conf_thres,
            'iou_thres': self.iou_thres,
            'max_det': self.max_det,
            'classes': self.classes,
            'agnostic_nms': self.agnostic_nms,
            'augment': self.augment,
            'visualize': self.visualize,
        }
        p.update(kwargs)

        im0 = img
        # Padded resize
        im = letterbox(im0, self.imgsz, self.model.stride, auto=self.model.pt)[0]
        # HWC to CHW, BGR to RGB
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        # Inference
        pred = self.model(im, augment=p['augment'], visualize=p['visualize'])

        # NMS
        pred = non_max_suppression(pred, p['conf_thres'], p['iou_thres'],
                                   p['classes'], p['agnostic_nms'], max_det=p['max_det'])

        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = [round(v.item()) for v in xyxy]
                    detections.append({
                        'class': self.names[int(cls)],
                        'conf': round(float(conf), 2),
                        'position': [x1, y1, x2 - x1, y2 - y1],
                    })
        return detections


# ============================================================
# 向后兼容：保留原函数名作为薄包装
# ============================================================

def loadmodel(weights, device='', imgsz=(640, 640), half=False, dnn=False, verbose=False):
    """加载模型（兼容旧接口）。返回 (model, imgsz, half)。"""
    det = YOLOv5Detector(weights, device=device, imgsz=imgsz, half=half, dnn=dnn, verbose=verbose)
    return det.model, det.imgsz, det.half


def detect(img, model, imgsz, device=None, conf_thres=0.30, iou_thres=0.45,
           max_det=1000, classes=None, agnostic_nms=False, augment=False,
           visualize=False, half=None):
    """单张图片推理（兼容旧接口）。"""
    det = YOLOv5Detector.__new__(YOLOv5Detector)
    det.model = model
    det.imgsz = imgsz
    det.device = device or model.device
    det.half = half if half is not None else getattr(model, 'fp16', False)
    det.names = model.names
    det.conf_thres = conf_thres
    det.iou_thres = iou_thres
    det.max_det = max_det
    det.classes = classes
    det.agnostic_nms = agnostic_nms
    det.augment = augment
    det.visualize = visualize
    return det.detect(img)


def detectdraw(frame, detection, color=(0, 255, 0), line_thickness=2):
    """绘制检测框（兼容旧接口）。"""
    YOLOv5BaseDetector.draw(frame, detection, color, line_thickness)
