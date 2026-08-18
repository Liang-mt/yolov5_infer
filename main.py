import argparse
from pathlib import Path

import cv2
import torch
import yaml

from main_detect import YOLOv5Detector
from onnxruntime_detect import YOLOv5OnnxRuntimeDetector
from opencv_detect import YOLOv5OpenCVDNN
from tensorrt_detect import YOLOv5TensorRTDetector

# 各后端默认权重
_DEFAULT_WEIGHTS = {
    'pytorch': './weights/yolov5s.pt',
    'onnxruntime': './weights/yolov5s.onnx',
    'opencv': './weights/yolov5s.onnx',
    'tensorrt': './weights/yolov5s.engine',
}


class YOLOv5Inference:
    """YOLOv5 多后端推理入口。

    Usage:
        app = YOLOv5Inference(backend='pytorch', source='./images/bus.jpg')
        app.run()
    """

    def __init__(self,
                 backend='pytorch',
                 weights='',
                 source='./images/bus.jpg',
                 imgsz=640,
                 conf_thres=0.30,
                 iou_thres=0.45,
                 device='auto',
                 data='',
                 view_img=True):
        self.backend = backend
        self.weights = weights or _DEFAULT_WEIGHTS[backend]
        self.source = source
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.data = data
        self.view_img = view_img
        self.detector = None

    @staticmethod
    def _load_names(data_path):
        """从 yaml 文件加载类别名，返回 {id: name} 字典。"""
        with open(data_path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        names = d.get('names', d.get('nc', {}))
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        return names

    def _create_detector(self):
        """根据后端创建对应的检测器。"""
        if not Path(self.weights).exists():
            raise FileNotFoundError(f"权重文件不存在: {self.weights}")

        names = self._load_names(self.data) if self.data else None
        sz = (self.imgsz, self.imgsz)

        if self.backend == 'pytorch':
            device = '0' if torch.cuda.is_available() else 'cpu'
            print(f"[PyTorch] device={device}")
            return YOLOv5Detector(self.weights, device=device, imgsz=sz,
                                  conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                                  names=names)

        elif self.backend == 'onnxruntime':
            det = YOLOv5OnnxRuntimeDetector(self.weights, imgsz=sz,
                                            conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                                            device=self.device, names=names)
            print(f"[ONNX Runtime] device={det.device} ({det.provider})")
            return det

        elif self.backend == 'opencv':
            det = YOLOv5OpenCVDNN(self.weights, imgsz=sz,
                                  conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                                  names=names)
            print(f"[OpenCV DNN] device={det.backend}")
            return det

        elif self.backend == 'tensorrt':
            det = YOLOv5TensorRTDetector(self.weights, imgsz=sz,
                                         conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                                         names=names)
            print(f"[TensorRT] device=CUDA (FP16)")
            return det

    def run(self):
        """执行推理主流程。"""
        if not Path(self.source).exists():
            raise FileNotFoundError(f"图片文件不存在: {self.source}")

        print(f"后端: {self.backend}")
        self.detector = self._create_detector()

        frame = cv2.imread(self.source)
        if frame is None:
            raise ValueError(f"图片读取失败: {self.source}")

        results = self.detector.detect(frame)
        for det in results:
            self.detector.draw(frame, det)

        if self.view_img:
            cv2.imshow(f'YOLOv5 ({self.backend})', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return results


def parse_opt():
    parser = argparse.ArgumentParser(description='YOLOv5 多后端推理')
    parser.add_argument('--backend', type=str, default='tensorrt', choices=['pytorch', 'onnxruntime', 'opencv', 'tensorrt'], help='推理后端')
    parser.add_argument('--weights', type=str, default='', help='权重文件（留空则按后端自动选择）')
    parser.add_argument('--source', type=str, default='./images/bus.jpg', help='输入图片路径')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图片尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.30, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU 阈值')
    parser.add_argument('--device', type=str, default='auto', help='设备: cuda / cpu / auto（仅 onnxruntime 后端生效）')
    parser.add_argument('--data', type=str, default='', help='数据集 yaml 文件（自定义类别名，留空用 COCO 80 类）')
    parser.add_argument('--view-img', action=argparse.BooleanOptionalAction, default=True, help='是否显示结果窗口（用 --no-view-img 关闭）')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        opt = parse_opt()
        app = YOLOv5Inference(
            backend=opt.backend,
            weights=opt.weights,
            source=opt.source,
            imgsz=opt.imgsz,
            conf_thres=opt.conf_thres,
            iou_thres=opt.iou_thres,
            device=opt.device,
            data=opt.data,
            view_img=opt.view_img,
        )
        app.run()
    except Exception as e:
        print(f"\n程序出错: {e}")
        exit(1)
