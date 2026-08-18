# YOLOv5 ONNX Runtime 推理
import os
import sys
from pathlib import Path
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from base_detector import YOLOv5BaseDetector

# 自动把 PyTorch 自带的 CUDA/cuDNN DLL 目录加入 PATH，
# 这样 onnxruntime-gpu 能直接复用 torch 的 CUDA 环境，无需单独安装 CUDA Toolkit
def _add_torch_cuda_to_path():
    try:
        import torch
        torch_lib = str(Path(torch.__file__).parent / 'lib')
        if torch_lib not in os.environ['PATH']:
            os.environ['PATH'] = torch_lib + os.pathsep + os.environ['PATH']
    except Exception:
        pass

_add_torch_cuda_to_path()
import onnxruntime as ort


class YOLOv5OnnxRuntimeDetector(YOLOv5BaseDetector):
    """基于 ONNX Runtime 的 YOLOv5 目标检测器。

    Usage:
        detector = YOLOv5OnnxRuntimeDetector('weights/yolov5s.onnx')
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
                 device='auto',
                 providers=None,
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
            device: 推理设备，'cuda' / 'cpu' / 'auto'(自动优先 CUDA)
            providers: 直接指定 ONNX Runtime providers（高级选项，设了则忽略 device）
            names: 类别名字典 {id: name}，None 则用 COCO 80 类
        """
        super().__init__(imgsz, conf_thres, iou_thres, max_det,
                         classes, agnostic_nms, names)

        # 根据 device 选择 providers
        if providers is None:
            has_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
            if device == 'cuda':
                if not has_cuda:
                    raise RuntimeError("ONNX Runtime 不支持 CUDA，请安装 onnxruntime-gpu 并确保 CUDA/cuDNN 在 PATH 中")
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif device == 'cpu':
                providers = ['CPUExecutionProvider']
            else:  # auto
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if has_cuda else ['CPUExecutionProvider']

        self.session = ort.InferenceSession(weights, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.provider = self.session.get_providers()[0]
        self.device = 'cuda' if 'CUDA' in self.provider else 'cpu'

        # warmup（CUDA 首次推理会初始化 kernel）
        if self.device == 'cuda':
            dummy = np.zeros((1, 3, self.imgsz[0], self.imgsz[1]), dtype=np.float32)
            self.session.run([self.output_name], {self.input_name: dummy})

    def detect(self, img, **kwargs):
        """单张图片推理。

        Args:
            img: BGR 图片 (numpy array)
            **kwargs: 临时覆盖参数，如 conf_thres=0.5

        Returns:
            list[dict]: [{'class': str, 'conf': float, 'position': [l,t,w,h]}, ...]
        """
        # 预处理
        im, _, _ = self.letterbox(img, self.imgsz)
        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        im = np.ascontiguousarray(im).astype(np.float32) / 255.0
        im = im[None]  # add batch dim

        # 推理
        pred = self.session.run([self.output_name], {self.input_name: im})[0]

        # 后处理
        return self.postprocess(pred[0], img.shape, **kwargs)
