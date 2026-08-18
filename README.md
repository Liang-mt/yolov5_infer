# YOLOv5-7.0 Revise — 多骨干网络改进版

基于 [YOLOv5 v7.0](https://github.com/ultralytics/yolov5) 的二次开发项目，集成了 **C2f、MobileNetV3、SE 注意力**三种自定义骨干网络/模块，并配套数据清理与模型精简工具，面向人脸检测及轻量化部署场景。

---

## 项目结构

```
yolov5-7.0_revise/
├── models/
│   ├── common.py                  # 核心模块（含 C2f / MobileNetV3 / SE 自定义实现）
│   ├── yolo.py                    # 模型解析与构建（已注册自定义模块）
│   ├── yolov5s-c2f.yaml           # C2f 骨干网络配置
│   ├── yolov5s-mobilenet.yaml     # MobileNetV3-Small 骨干网络配置
│   ├── yolov5s-se.yaml            # SE 注意力模块配置
│   ├── yolov5n.yaml / yolov5s.yaml  # 原版基线配置
│   └── hub/                       # 其他变体配置（Ghost、Transformer、P6 等）
├── utils/                         # YOLOv5 原工具集（损失、数据增强、指标等）
├── data/                          # 数据集配置（COCO、VOC、VisDrone 等）
├── weights/                       # 训练权重与导出模型
│   ├── best_mask.pt               # 口罩检测最佳权重
│   ├── best_mask.onnx             # ONNX 部署模型
│   └── yolov5n.pt / yolov5s.pt    # 预训练基线
├── dataset_revise.py              # 数据集损坏标签清理工具
├── conver_pt1_2.py                # 模型深度精简工具（FP16 + Fuse + 去冗余）
├── train.py                       # 训练入口
├── detect.py                      # 推理入口
├── val.py                         # 验证入口
├── export.py                      # 模型导出入口
├── coco128.yaml                   # 快速测试数据集配置
└── runs/train/                    # 训练实验记录（exp ~ exp15）
```

---

## 环境要求

```bash
pip install -r requirements.txt
```

- Python >= 3.8
- PyTorch >= 1.7
- torchvision（MobileNetV3 预训练权重依赖）
- CUDA 可选（CPU 可运行但训练较慢）

> **注意**：使用 MobileNetV3 骨干时，首次运行会自动下载 ImageNet 预训练权重，需保持网络连接。

---

## 自定义模块说明

### 1. C2f — YOLOv8 风格 CSP 瓶颈结构

文件：`models/common.py` → `class C2f` / `class C2fBottleneck`

将 YOLOv5 的 C3 模块替换为 YOLOv8 的 C2f，通过更细粒度的通道分流和拼接，在相同参数量下获得更丰富的梯度流。

配置：`models/yolov5s-c2f.yaml`
- Backbone 使用 C2f，Head 保留 C3
- 适用场景：追求更高检测精度，可接受少量计算量增加

### 2. MobileNetV3-Small — 轻量化骨干网络

文件：`models/common.py` → `class MobileNetV3`

使用 torchvision 提供的 MobileNetV3-Small 作为特征提取骨干，将 features 分为三段输出（P3/P4/P5），替换 YOLOv5 原有的 Conv+C3 堆叠。

配置：`models/yolov5s-mobilenet.yaml`
- 三段输出通道：24 → 48 → 576
- 内置 ImageNet 预训练权重
- 适用场景：边缘设备、移动端部署，追求极致轻量化

### 3. SE — 通道注意力模块

文件：`models/common.py` → `class SE`

Squeeze-and-Excitation 通道注意力，通过全局平均池化 + 两层全连接学习通道权重，增强重要特征、抑制无关特征。

配置：`models/yolov5s-se.yaml`
- 在 Backbone 末尾（P5 层）插入 SE，压缩比 ratio=2
- 适用场景：细粒度检测、通道间相关性强的场景

---

## 快速开始

### 训练

```bash
# 基线模型（YOLOv5n）
python train.py --cfg models/yolov5n.yaml --weights yolov5n.pt --data coco128.yaml --epochs 100 --batch-size 16

# C2f 改进版
python train.py --cfg models/yolov5s-c2f.yaml --weights yolov5n.pt --data coco128.yaml --epochs 100 --batch-size 16

# MobileNetV3 轻量化版
python train.py --cfg models/yolov5s-mobilenet.yaml --weights yolov5n.pt --data coco128.yaml --epochs 100 --batch-size 16

# SE 注意力版
python train.py --cfg models/yolov5s-se.yaml --weights yolov5n.pt --data coco128.yaml --epochs 100 --batch-size 16
```

常用参数：
- `--cfg`：模型配置文件
- `--weights`：预训练权重（自定义结构与权重不匹配时，仅加载可匹配层）
- `--data`：数据集配置
- `--epochs`：训练轮数
- `--batch-size`：批大小
- `--imgsz`：输入图像尺寸，默认 640
- `--device`：GPU 设备，如 `0` / `0,1` / `cpu`

### 推理

```bash
python detect.py --weights weights/best_mask.pt --source 1.mp4 --conf-thres 0.25
```

- `--source`：支持图片、视频、文件夹、摄像头（`0`）、RTSP 流
- `--conf-thres`：置信度阈值
- `--iou-thres`：NMS IoU 阈值
- `--save-txt`：保存检测结果为 txt

### 验证

```bash
python val.py --weights weights/best_mask.pt --data coco128.yaml
```

### 导出 ONNX

```bash
python export.py --weights weights/best_mask.pt --include onnx --imgsz 640
```

---

## 工具脚本

### 1. 数据集清理工具 `dataset_revise.py`

用于清理 WIDERFace 等数据集转换为 YOLO 格式后产生的**损坏标签**（负坐标、越界值）。

```bash
python dataset_revise.py
```

工作流程：
1. 从训练日志中正则提取有问题的图片路径
2. 自动匹配对应的标签文件
3. 确认后批量删除图片+标签

> 当前日志内容和路径内置在脚本中，使用前需修改 `log_content` 和路径正则。

### 2. 模型深度精简工具 `conver_pt1_2.py`

对训练好的 `.pt` 权重进行推理优化，产出体积更小、速度更快的部署模型。

```bash
# 直接清理模式（无需 cfg）
python conver_pt1_2.py --input weights/best_mask.pt --output weights/best_mask_.pt

# CFG 重建模式（从 yaml 重建纯净模型再加载权重）
python conver_pt1_2.py --input weights/best_mask.pt --output weights/best_mask_.pt --cfg models/yolov5s.yaml --progress
```

优化措施：
- 剔除训练辅助参数（aux head、optimizer 状态等）
- Conv + BatchNorm 层融合（Fuse）
- FP16 半精度转换
- 紧凑序列化，减少元数据开销
- CFG 模式下从配置重建模型，彻底清除训练历史

---

## 训练实验记录

`runs/train/` 下共 15 组实验：

| 实验 | 模型配置 | Epochs | 说明 |
|------|---------|--------|------|
| exp | yolov5s-mobilenet | 1500 | MobileNetV3 骨干长时训练 |
| exp2 | yolov5s-c2f | 10 | C2f 骨干快速验证 |
| exp3-4 | yolov5s-se | 10 | SE 注意力快速验证 |
| exp5-15 | yolov5n | 10-100 | 基线模型对比与调参 |

每组实验目录包含：
- `opt.yaml`：完整训练参数
- `hyp.yaml`：超参数配置
- `weights/`：best.pt / last.pt
- `results.csv`：训练指标曲线
- `confusion_matrix.png`：混淆矩阵

---

## 数据集配置

项目支持多种数据集，配置文件位于 `data/`：

| 配置文件 | 数据集 | 类别数 |
|---------|--------|--------|
| `coco.yaml` | COCO | 80 |
| `coco128.yaml` | COCO128（快速测试） | 80 |
| `VOC.yaml` | PASCAL VOC | 20 |
| `VisDrone.yaml` | 无人机航拍 | 10 |
| `Argoverse.yaml` | 自动驾驶 | 15 |
| `xView.yaml` | 卫星图像 | 60 |

自定义数据集需编写 yaml，指定 `train` / `val` 路径和 `names` 类别列表。

---

## 注意事项

1. **自定义模型与预训练权重**：使用 `--cfg` 指定自定义结构（MobileNetV3/C2f/SE）时，`--weights` 传入的标准权重仅能加载部分匹配层，其余层随机初始化，建议适当增加训练轮数。

2. **MobileNetV3 预训练权重**：首次使用需联网下载，torchvision 版本需兼容。离线环境可提前下载权重放入缓存目录。

3. **SE 模块 API**：当前使用 `F.sigmoid`，在较新版本 PyTorch 中会产生 DeprecationWarning，功能正常，可替换为 `torch.sigmoid`。

4. **数据集路径**：根目录 `coco128.yaml` 与 `data/coco128.yaml` 为两个独立配置，路径指向不同，使用时注意区分。

5. **模型精简后验证**：`conver_pt1_2.py` 输出的精简模型建议用 `val.py` 跑一次验证，确认精度无损失后再部署。

---

## 参考

- [YOLOv5 官方仓库](https://github.com/ultralytics/yolov5)
- [YOLOv8 C2f 模块](https://github.com/ultralytics/ultralytics)
- [MobileNetV3 论文](https://arxiv.org/abs/1905.02244)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [WIDERFace 数据集](http://shuoyang1213.me/WIDERFACE/)
