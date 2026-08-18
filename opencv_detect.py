# YOLOv5 OpenCV DNN 推理
import sys
from pathlib import Path
import numpy as np
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from base_detector import YOLOv5BaseDetector


class YOLOv5OpenCVDNN(YOLOv5BaseDetector):
    """基于 OpenCV DNN 模块的 YOLOv5 目标检测器。

    无需 PyTorch / ONNX Runtime，仅依赖 OpenCV，适合轻量部署。

    Usage:
        detector = YOLOv5OpenCVDNN('weights/yolov5s.onnx')
        results = detector.detect(frame)
        for det in results:
            detector.draw(frame, det)
    """

    def __init__(self,
                 weights,
                 imgsz=(640, 640),
                 conf_thres=0.30,
                 iou_thres=0.45,
                 max_det=1000,
                 classes=None,
                 agnostic_nms=False,
                 use_cuda=True,
                 names=None):
        """
        Args:
            weights: ONNX 模型文件路径
            imgsz: 输入图片尺寸 (h, w)
            conf_thres: 置信度阈值
            iou_thres: NMS IOU 阈值
            max_det: 单图最大检测数
            classes: 只保留指定类别 (None=全部)
            agnostic_nms: 类别无关 NMS
            use_cuda: 是否尝试 CUDA 后端（不支持时自动回退 CPU）
            names: 类别名字典 {id: name}，None 则用 COCO 80 类
        """
        super().__init__(imgsz, conf_thres, iou_thres, max_det,
                         classes, agnostic_nms, names)

        # 加载 ONNX 模型
        self.net = cv2.dnn.readNetFromONNX(weights)

        # 检测 OpenCV 是否编译了 CUDA 后端
        cuda_built = 'CUDA' in cv2.getBuildInformation()

        if use_cuda and cuda_built:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.backend = 'CUDA'
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            if use_cuda and not cuda_built:
                self.backend = 'CPU (CUDA 不可用，已回退)'
            else:
                self.backend = 'CPU'

    def detect(self, img, **kwargs):
        """单张图片推理。

        Args:
            img: BGR 图片 (numpy array)
            **kwargs: 临时覆盖参数

        Returns:
            list[dict]: [{'class': str, 'conf': float, 'position': [l,t,w,h]}, ...]
        """
        # 预处理：letterbox + blob
        im, _, _ = self.letterbox(img, self.imgsz)
        blob = cv2.dnn.blobFromImage(im, 1 / 255.0, (self.imgsz[1], self.imgsz[0]),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)

        # 推理，输出 [1, 25200, 85]
        pred = self.net.forward()[0]

        # 后处理
        return self.postprocess(pred, img.shape, **kwargs)
