# YOLOv5-FasterNet 目标检测

基于 [YOLOv5 v6.0](https://github.com/ultralytics/yolov5) 改进的轻量级目标检测项目，将骨干网络替换为微软 [FasterNet](https://github.com/JierunChen/FasterNet)（Partial Convolution），在保持高精度的同时大幅减少参数量和计算量。

## 模型特点

| 项目 | 数值 |
|------|------|
| 骨干网络 | FasterNet-t0（PConv 部分卷积） |
| 参数量 | 3.3M |
| 计算量 | 7.5 GFLOPs |
| 模型大小 | 6.9 MB（FP32） |
| 训练数据集 | COCO128（128 张） |
| 训练轮次 | 1000 epochs |
| mAP@0.5 | 0.832 |
| mAP@0.5:0.95 | 0.584 |

FasterNet 的核心是 **Partial Convolution（PConv）**：只对部分输入通道做卷积，其余通道直接跳过，减少冗余计算和内存访问，在 CPU/边缘设备上推理速度更快。

## 环境要求

- Python 3.9+
- PyTorch 2.0+（已在 2.7.0+cu128 上验证）
- CUDA（GPU 推理/训练，CPU 亦可）

### 安装依赖

```bash
pip install -r requirements.txt
pip install onnx onnxruntime  # 如需 ONNX 导出/推理
```

> 本项目已修复 PyTorch 2.6+、Pillow 10+、protobuf 6+、timm 0.9+、albumentations 2+ 等新版依赖的兼容性问题，详见 [修改优化文档.md](修改优化文档.md)。

## 快速开始

### 推理检测

```bash
python detect.py --weights exp_1000_fasternrt0/weight/best.pt --source data/images/bus.jpg
```

常用参数：
- `--source 0`：摄像头实时检测
- `--source path/to/video.mp4`：视频检测
- `--conf-thres 0.5`：置信度阈值
- `--device 0`：指定 GPU，`--device cpu` 使用 CPU
- `--nosave`：不保存结果图片

### 模型验证

```bash
python val.py --weights exp_1000_fasternrt0/weight/best.pt --data data/coco128.yaml
```

### 训练

```bash
python train.py --weights '' --cfg models/yolov5-custom.yaml --data data/coco128.yaml --epochs 100 --batch-size 16
```

- `--weights ''`：从头训练（会自动加载 FasterNet-t0 ImageNet 预训练权重）
- `--weights exp_1000_fasternrt0/weight/best.pt`：从已有权重继续训练
- 训练结果保存在 `runs/train/exp/` 目录

### 导出 ONNX

```bash
python export.py --weights exp_1000_fasternrt0/weight/best.pt --include onnx --img 640
```

导出后可用 Netron（https://netron.app ）查看模型结构。

### 模型压缩（FP16）

```bash
python conver_pt.py --input exp_1000_fasternrt0/weight/best.pt --output compressed.pt
```

将 FP32 模型转为 FP16 半精度，体积减半、GPU 推理加速，精度损失极小。

### 性能基准测试

```bash
python benchmarks.py --weights exp_1000_fasternrt0/weight/best.pt
```

## 项目结构

```
yolov5_infer-1.0/
├── train.py                    # 训练入口
├── detect.py                   # 推理检测入口
├── val.py                      # 模型验证入口
├── export.py                   # 模型导出（ONNX/TorchScript 等）
├── conver_pt.py                # 模型压缩（FP32→FP16）
├── benchmarks.py               # 性能基准测试
├── yolo2coco.py                # YOLO 标注格式转 COCO 格式
├── hubconf.py                  # PyTorch Hub 配置
├── requirements.txt            # 依赖列表
│
├── models/
│   ├── yolo.py                 # 模型构建（解析 yaml、创建网络）
│   ├── common.py               # 通用模块（Conv/C3/SPPF/Detect 等）
│   ├── experimental.py         # 实验性功能（attempt_load 等）
│   ├── fasternet.py            # FasterNet 骨干网络实现
│   ├── mobilenetv4.py          # MobileNetV4 骨干（备选）
│   ├── yolov5-custom.yaml      # 本项目模型配置（FasterNet-t0 骨干）
│   └── faster_cfg/             # FasterNet 各规格配置（t0/t1/t2/s/m/l）
│
├── utils/
│   ├── general.py              # 通用工具（下载、检查、NMS、坐标转换等）
│   ├── torch_utils.py          # PyTorch 工具（设备选择、模型信息等）
│   ├── plots.py                # 绘图工具（检测框、训练曲线等）
│   ├── metrics.py              # 评估指标（mAP/IoU/混淆矩阵）
│   ├── loss.py                 # 损失函数
│   ├── dataloaders.py          # 数据加载
│   ├── augmentations.py        # 数据增强
│   ├── autoanchor.py           # 自动锚框聚类
│   ├── autobatch.py            # 自动批次大小
│   ├── callbacks.py            # 训练回调
│   ├── downloads.py            # 下载工具
│   ├── activations.py          # 激活函数
│   ├── segment/                # 分割任务支持
│   ├── loggers/                # 日志记录（TensorBoard/W&B/ClearML/Comet）
│   └── aws/                    # AWS 训练支持
│
├── data/
│   ├── coco128.yaml            # COCO128 数据集配置
│   ├── coco.yaml               # COCO 数据集配置
│   ├── hyps/                   # 超参数配置
│   ├── images/                 # 示例图片（bus.jpg, zidane.jpg）
│   └── scripts/                # 数据集下载脚本
│
├── dataset/                    # 项目自带验证集（128 张 COCO val 图片）
│   ├── images/val/
│   └── labels/val/
│
├── exp_1000_fasternrt0/        # 1000 epoch 训练结果
│   ├── weight/
│   │   ├── best.pt             # 最佳权重（6.9 MB）
│   │   ├── last.pt             # 最后一轮权重
│   │   └── best.onnx           # 导出的 ONNX 模型（13 MB）
│   ├── results.csv             # 训练指标记录
│   ├── results.png             # 训练曲线图
│   ├── confusion_matrix.png    # 混淆矩阵
│   ├── PR_curve.png            # PR 曲线
│   └── ...
│
└── fasternet_t0-epoch.281-val_acc1.71.9180.pth  # FasterNet-t0 ImageNet 预训练权重
```

## 数据集格式

使用 YOLO 格式标注，目录结构：

```
dataset/
├── images/
│   ├── train/
│   │   ├── 0001.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── 0001.txt    # 每行: class_id x_center y_center width height（归一化）
    │   └── ...
    └── val/
        └── ...
```

自定义数据集需创建对应的 `.yaml` 配置文件（参考 `data/coco128.yaml`）。

## 技术说明

### FasterNet 骨干替换

模型配置 `models/yolov5-custom.yaml` 中，骨干部分用 `fasternet_t0` 替换了原版 YOLOv5 的 CSPDarknet，输出 3 个尺度的特征图（P3/P4/P5）接入 YOLOv5 Head。FasterNet-t0 的 ImageNet 预训练权重（top-1 71.918%）在训练时自动加载。

### 兼容性修复

本项目基于 YOLOv5 v6.0 代码，已适配 2025-2026 年的新版 Python 生态（PyTorch 2.7、Pillow 10、protobuf 6、timm 1.x、albumentations 2.x 等），所有核心功能均可直接运行。

## License

GPL-3.0（继承 YOLOv5 协议）
