# YOLOv5 Inference Toolkit v2.0

基于 YOLOv5 7.0 的多后端推理封装，支持 **PyTorch / ONNX Runtime / TensorRT / OpenCV DNN** 四种推理后端，统一接口，一键切换。

## 特性

- **四后端统一接口**：`detect()` / `draw()` / `__call__` 完全一致，切换后端只改一个参数
- **GPU 加速**：PyTorch CUDA、ONNX Runtime CUDA、TensorRT FP16 均已适配
- **自动环境兼容**：PyTorch 2.6+ 权重加载、TensorRT 10.x API、CUDA DLL 自动注入
- **公共基类**：所有后端继承统一基类，预处理/后处理/绘制逻辑只写一份
- **自定义模型支持**：可传入 yaml 或字典指定类别名，支持自训练模型
- **资源安全**：TensorRT 支持 `close()` 显式释放 GPU 显存
- **向后兼容**：保留 YOLOv5 官方 `detect.py` / `export.py` 完整功能

## 后端对比

| 后端 | 依赖 | GPU 支持 | 速度 (RTX 5060, 640x640, 100帧平均) | 权重格式 |
|------|------|---------|--------------------------------------|---------|
| PyTorch | torch | CUDA | 14.3 ms / 69.9 FPS | .pt |
| ONNX Runtime | onnxruntime-gpu | CUDA / CPU | 14.6 ms / 68.4 FPS (GPU) · 74.1 ms / 13.5 FPS (CPU) | .onnx |
| TensorRT | tensorrt, pycuda | CUDA FP16 | 13.7 ms / 72.8 FPS | .engine |
| OpenCV DNN | opencv-python | 仅 CPU | 562.4 ms / 1.8 FPS | .onnx (需简化) |

> 四后端在 bus.jpg 上检测结果完全一致：4 person + 1 bus。

## 环境要求

- Python 3.9+
- CUDA 12.x + cuDNN 9.x（GPU 推理）
- 依赖见 `requirements.txt`

### 安装

```bash
# 创建虚拟环境
conda create -n yolov5 python=3.9
conda activate yolov5

# 安装 PyTorch（按你的 CUDA 版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 安装核心依赖
pip install -r requirements.txt

# TensorRT 后端（可选）
pip install tensorrt pycuda
```

## 快速开始

```bash
# PyTorch 推理
python main.py --backend pytorch

# ONNX Runtime GPU 推理
python main.py --backend onnxruntime --device cuda

# TensorRT 推理
python main.py --backend tensorrt

# OpenCV DNN 推理（纯 CPU，无需 torch）
python main.py --backend opencv

# 不弹窗
python main.py --backend pytorch --no-view-img
```

### 速度测试

```bash
# 默认 100 帧
python benchmark.py

# 自定义帧数和尺寸
python benchmark.py --frames 200 --imgsz 416
```

## 命令行参数

```
--backend       推理后端: pytorch / onnxruntime / opencv / tensorrt（默认 pytorch）
--weights       权重文件路径（留空按后端自动选择）
--source        输入图片路径（默认 ./images/bus.jpg）
--imgsz         输入图片尺寸（默认 640，自动对齐到 32 的倍数）
--conf-thres    置信度阈值（默认 0.30）
--iou-thres     NMS IOU 阈值（默认 0.45）
--device        设备: cuda / cpu / auto（仅 onnxruntime 后端，默认 auto）
--data          数据集 yaml 文件（自定义类别名，留空用 COCO 80 类）
--view-img      显示结果窗口（默认开启，用 --no-view-img 关闭）
```

示例：

```bash
# 自定义图片和阈值
python main.py --backend onnxruntime --source ./test.jpg --conf-thres 0.5

# ONNX Runtime 强制 CPU
python main.py --backend onnxruntime --device cpu

# 自训练模型（指定类别名）
python main.py --backend tensorrt --weights ./weights/best.engine --data ./data/custom.yaml
```

## Python API

### 基础用法

```python
import cv2
from main_detect import YOLOv5Detector

# 加载模型
detector = YOLOv5Detector('./weights/yolov5s.pt', device='0')

# 推理
frame = cv2.imread('./images/bus.jpg')
results = detector.detect(frame)

for det in results:
    print(det['class'], det['conf'], det['position'])
    # person 0.90 [221, 407, 125, 468]
    detector.draw(frame, det)

cv2.imshow('result', frame)
cv2.waitKey(0)
```

### 切换后端

```python
# ONNX Runtime
from onnxruntime_detect import YOLOv5OnnxRuntimeDetector
detector = YOLOv5OnnxRuntimeDetector('./weights/yolov5s.onnx', device='cuda')

# TensorRT
from tensorrt_detect import YOLOv5TensorRTDetector
detector = YOLOv5TensorRTDetector('./weights/yolov5s.engine')

# OpenCV DNN
from opencv_detect import YOLOv5OpenCVDNN
detector = YOLOv5OpenCVDNN('./weights/yolov5s.onnx')

# 接口完全一致
results = detector.detect(frame)
```

### 简写调用

```python
# __call__ 等价于 detect()
results = detector(frame)
```

### 临时覆盖参数

```python
# 本次推理用更高阈值，不影响实例默认值
results = detector.detect(frame, conf_thres=0.6, classes=[0, 5], max_det=10)
```

### 自定义类别名

```python
# PyTorch 后端默认从 .pt 模型读取类别名
# 其他后端可传入 names 字典，或通过 --data 指定 yaml 文件
names = {0: 'cat', 1: 'dog'}
detector = YOLOv5OnnxRuntimeDetector('./weights/best.onnx', names=names)
```

### TensorRT 资源释放

```python
detector = YOLOv5TensorRTDetector('./weights/yolov5s.engine')
results = detector.detect(frame)
detector.close()  # 显式释放 GPU 显存（不调用时 __del__ 兜底）
```

> TensorRT engine 是固定尺寸的，`imgsz` 必须与导出 engine 时一致，否则会报错提示。

### 使用 YOLOv5Inference 类

```python
from main import YOLOv5Inference

app = YOLOv5Inference(
    backend='tensorrt',
    source='./images/bus.jpg',
    conf_thres=0.5,
    view_img=True,
)
results = app.run()
```

## 模型导出

### 导出 ONNX

```bash
python export.py --weights weights/yolov5s.pt --include onnx --imgsz 640 --device 0
```

> OpenCV DNN 后端需要简化后的 ONNX（原始 ONNX 含不支持的 Floor 节点）：
> ```bash
> pip install onnx onnx-simplifier
> python -c "import onnx; from onnxsim import simplify; m=onnx.load('weights/yolov5s.onnx'); m,_=simplify(m); onnx.save(m,'weights/yolov5s.onnx')"
> ```

### 导出 TensorRT Engine

```bash
# 默认：weights/yolov5s.onnx -> weights/yolov5s.engine，FP16，640x640
python export_tensorrt.py

# 完整参数
python export_tensorrt.py \
    --weights weights/yolov5s.onnx \
    --output weights/yolov5s.engine \
    --imgsz 640 \
    --workspace 4 \
    --fp16
```

参数说明：

```
--weights     输入 ONNX 模型路径（默认 weights/yolov5s.onnx）
--output      输出 engine 路径（留空与 ONNX 同名）
--imgsz       输入尺寸（默认 640）
--workspace   GPU workspace 大小 GB（默认 4）
--fp16        启用 FP16（默认开启，--no-fp16 关闭）
```

> TensorRT engine 与 GPU 架构、TensorRT 版本绑定，换机器需重新导出。

## 检测结果格式

`detect()` 返回 `list[dict]`，每个元素：

```python
{
    'class': 'person',           # 类别名
    'conf': 0.90,                # 置信度
    'position': [221, 407, 125, 468]  # [left, top, width, height]
}
```

## 项目结构

```
yolov5_infer-master/
├── main.py                    # 主入口（YOLOv5Inference 类 + argparse）
├── benchmark.py               # 多后端推理速度基准测试
├── base_detector.py           # 公共基类（letterbox/NMS/postprocess/draw）
├── main_detect.py             # PyTorch 后端（YOLOv5Detector）
├── onnxruntime_detect.py      # ONNX Runtime 后端（YOLOv5OnnxRuntimeDetector）
├── tensorrt_detect.py         # TensorRT 后端（YOLOv5TensorRTDetector）
├── opencv_detect.py           # OpenCV DNN 后端（YOLOv5OpenCVDNN）
├── detect.py                  # YOLOv5 官方检测脚本
├── export.py                  # YOLOv5 官方模型导出
├── export_tensorrt.py         # TensorRT engine 导出脚本
├── models/                    # YOLOv5 模型定义
├── utils/                     # YOLOv5 工具函数
├── data/coco128.yaml          # COCO 类别配置
├── weights/                   # 权重文件目录
│   ├── yolov5s.pt
│   ├── yolov5s.onnx
│   └── yolov5s.engine
├── images/                    # 测试图片
├── 修改优化文档.md             # 版本修改记录
└── requirements.txt
```

## 技术说明

- **PyTorch 2.6+ 兼容**：自动使用 `weights_only=False` 加载旧版 YOLOv5 权重
- **静默加载**：默认屏蔽 YOLOv5 的设备信息/Fusing layers/模型摘要输出
- **CUDA DLL 自动注入**：ONNX Runtime 和 TensorRT 自动复用 PyTorch 自带的 CUDA/cuDNN DLL，无需单独安装 CUDA Toolkit
- **imgsz 自动对齐**：输入尺寸非 32 倍数时自动调整并提示
- **warmup 预热**：模型加载时自动预热一次（PyTorch/ONNX Runtime CUDA/TensorRT），首帧不卡顿
- **标签智能定位**：检测框贴顶时标签自动画到框内，不会画出图片边界

## License

AGPL-3.0（同 YOLOv5）
