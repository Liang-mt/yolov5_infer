# YOLOv5 TensorRT 推理
import os
import sys
from pathlib import Path
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from base_detector import YOLOv5BaseDetector

# 自动把 PyTorch 自带的 CUDA/cuDNN/TensorRT DLL 目录加入 PATH
def _add_torch_lib_to_path():
    try:
        import torch
        torch_lib = str(Path(torch.__file__).parent / 'lib')
        if torch_lib not in os.environ['PATH']:
            os.environ['PATH'] = torch_lib + os.pathsep + os.environ['PATH']
    except Exception:
        pass

_add_torch_lib_to_path()
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class YOLOv5TensorRTDetector(YOLOv5BaseDetector):
    """基于 TensorRT 的 YOLOv5 目标检测器。

    Usage:
        detector = YOLOv5TensorRTDetector('weights/yolov5s.engine')
        results = detector.detect(frame)
        for det in results:
            detector.draw(frame, det)
        detector.close()  # 显式释放 GPU 资源
    """

    def __init__(self,
                 weights,
                 imgsz=(640, 640),
                 conf_thres=0.30,
                 iou_thres=0.45,
                 max_det=1000,
                 classes=None,
                 agnostic_nms=False,
                 names=None):
        """
        Args:
            weights: TensorRT engine 文件路径 (.engine)
            imgsz: 输入图片尺寸 (h, w)，需与导出 engine 时一致
            conf_thres: 置信度阈值
            iou_thres: NMS IOU 阈值
            max_det: 单图最大检测数
            classes: 只保留指定类别 (None=全部)
            agnostic_nms: 类别无关 NMS
            names: 类别名字典 {id: name}，None 则用 COCO 80 类
        """
        super().__init__(imgsz, conf_thres, iou_thres, max_det,
                         classes, agnostic_nms, names)

        # 加载 engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(weights, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"TensorRT engine 加载失败: {weights}\n"
                "可能原因: engine 文件损坏、TensorRT 版本不兼容、或不是有效的 engine 文件。\n"
                "请用 export_tensorrt.py 重新导出。"
            )
        self.context = self.engine.create_execution_context()

        # 获取输入输出 tensor 名称
        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        # 检查 engine 输入尺寸是否与 imgsz 匹配
        engine_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if len(engine_shape) == 4 and engine_shape[2:] != (self.imgsz[0], self.imgsz[1]):
            raise RuntimeError(
                f"engine 输入尺寸 {engine_shape[2:]} 与 imgsz={self.imgsz} 不匹配，\n"
                f"请用 --imgsz {engine_shape[2]} 或重新导出 engine。"
            )

        # 设置输入形状
        self.context.set_input_shape(self.input_name, (1, 3, self.imgsz[0], self.imgsz[1]))

        # 分配输入输出缓冲区
        self.input_h = np.empty((1, 3, self.imgsz[0], self.imgsz[1]), dtype=np.float32)
        self.output_h = np.empty(self._get_output_shape(), dtype=np.float32)
        self.input_d = cuda.mem_alloc(self.input_h.nbytes)
        self.output_d = cuda.mem_alloc(self.output_h.nbytes)
        self.stream = cuda.Stream()

        # TensorRT 10.x: 用 set_tensor_address 绑定设备地址
        self.context.set_tensor_address(self.input_name, int(self.input_d))
        self.context.set_tensor_address(self.output_name, int(self.output_d))

        # warmup（TensorRT 首次推理会初始化 CUDA kernel）
        self._warmup()

    def _get_output_shape(self):
        """获取输出张量形状。"""
        shape = self.context.get_tensor_shape(self.output_name)
        if shape[0] == -1:
            n = (self.imgsz[0] // 8) * (self.imgsz[1] // 8) * 3 + \
                (self.imgsz[0] // 16) * (self.imgsz[1] // 16) * 3 + \
                (self.imgsz[0] // 32) * (self.imgsz[1] // 32) * 3
            return (1, n, 85)
        return tuple(shape)

    def _warmup(self):
        """预热一次推理，避免首帧延迟。"""
        dummy = np.zeros((self.imgsz[0], self.imgsz[1], 3), dtype=np.uint8)
        im, _, _ = self.letterbox(dummy, self.imgsz)
        im = im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        self.input_h[0] = np.ascontiguousarray(im)
        cuda.memcpy_htod_async(self.input_d, self.input_h, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_h, self.output_d, self.stream)
        self.stream.synchronize()

    def detect(self, img, **kwargs):
        """单张图片推理。

        Args:
            img: BGR 图片 (numpy array)
            **kwargs: 临时覆盖参数

        Returns:
            list[dict]: [{'class': str, 'conf': float, 'position': [l,t,w,h]}, ...]
        """
        # 预处理
        im, _, _ = self.letterbox(img, self.imgsz)
        im = im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        im = np.ascontiguousarray(im)
        self.input_h[0] = im

        # 推理：H2D -> execute -> D2H
        cuda.memcpy_htod_async(self.input_d, self.input_h, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_h, self.output_d, self.stream)
        self.stream.synchronize()

        # 后处理
        return self.postprocess(self.output_h[0], img.shape, **kwargs)

    def close(self):
        """显式释放 GPU 内存和 TensorRT 资源。"""
        if hasattr(self, 'input_d') and self.input_d:
            self.input_d.free()
            self.input_d = None
        if hasattr(self, 'output_d') and self.output_d:
            self.output_d.free()
            self.output_d = None
        self.context = None
        self.engine = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
