"""YOLOv5 ONNX -> TensorRT engine 导出脚本。

Usage:
    python export_tensorrt.py                                        # 默认参数
    python export_tensorrt.py --weights weights/yolov5s.onnx --imgsz 640
    python export_tensorrt.py --weights weights/best.onnx --output weights/best.engine --no-fp16
"""
import argparse
import os
from pathlib import Path

import torch

# 把 torch lib 加入 PATH（复用 CUDA/cuDNN DLL）
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['PATH'] = torch_lib + os.pathsep + os.environ['PATH']

import tensorrt as trt


def parse_opt():
    parser = argparse.ArgumentParser(description='YOLOv5 ONNX -> TensorRT engine 导出')
    parser.add_argument('--weights', type=str, default='weights/yolov5s.onnx', help='输入 ONNX 模型路径')
    parser.add_argument('--output', type=str, default='', help='输出 engine 路径（留空则与 ONNX 同名）')
    parser.add_argument('--imgsz', type=int, default=640, help='输入尺寸（默认 640）')
    parser.add_argument('--workspace', type=int, default=4, help='GPU workspace 大小 GB（默认 4）')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True, help='启用 FP16（默认开启，--no-fp16 关闭）')
    return parser.parse_args()


def main():
    opt = parse_opt()

    if not Path(opt.weights).exists():
        raise FileNotFoundError(f"ONNX 文件不存在: {opt.weights}")

    output = opt.output or str(Path(opt.weights).with_suffix('.engine'))

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    print(f'解析 ONNX: {opt.weights}')
    with open(opt.weights, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError('ONNX 解析失败')

    # 检查输入尺寸
    input_tensor = network.get_input(0)
    print(f'模型输入: {input_tensor.name}, shape: {input_tensor.shape}')

    print(f'构建 engine (imgsz={opt.imgsz}, workspace={opt.workspace}GB, fp16={opt.fp16})...')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, opt.workspace << 30)
    if opt.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print('启用 FP16')
    elif opt.fp16:
        print('当前 GPU 不支持 FP16，使用 FP32')

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('engine 构建失败')

    with open(output, 'wb') as f:
        f.write(serialized)

    size_mb = os.path.getsize(output) / 1e6
    print(f'导出成功: {output} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
