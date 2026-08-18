"""YOLOv5 多后端推理速度基准测试。

Usage:
    python benchmark.py [--frames 100] [--imgsz 640]
"""
import argparse
import time
import cv2


def benchmark(name, detector, img, frames):
    # warmup
    for _ in range(10):
        detector.detect(img)

    # 计时
    t0 = time.perf_counter()
    for _ in range(frames):
        detector.detect(img)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / frames * 1000
    fps = frames / (t1 - t0)
    print(f"  {name:25s} {avg_ms:7.1f} ms/帧   {fps:6.1f} FPS")
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description='YOLOv5 多后端推理速度测试')
    parser.add_argument('--frames', type=int, default=100, help='测试帧数（默认 100）')
    parser.add_argument('--imgsz', type=int, default=640, help='输入尺寸（默认 640）')
    parser.add_argument('--source', type=str, default='./images/bus.jpg', help='测试图片')
    opt = parser.parse_args()

    img = cv2.imread(opt.source)
    assert img is not None, f"图片读取失败: {opt.source}"

    print(f"测试图片: {opt.source} ({img.shape[1]}x{img.shape[0]})")
    print(f"输入尺寸: {opt.imgsz}x{opt.imgsz}")
    print(f"测试帧数: {opt.frames}（含 10 帧 warmup，不计入）")
    print("-" * 55)

    results = {}

    # PyTorch CUDA
    try:
        from main_detect import YOLOv5Detector
        det = YOLOv5Detector('./weights/yolov5s.pt', device='0', imgsz=(opt.imgsz, opt.imgsz))
        results['PyTorch CUDA'] = benchmark('PyTorch CUDA', det, img, opt.frames)
    except Exception as e:
        print(f"  PyTorch CUDA             失败: {e}")

    # ONNX Runtime CUDA
    try:
        from onnxruntime_detect import YOLOv5OnnxRuntimeDetector
        det = YOLOv5OnnxRuntimeDetector('./weights/yolov5s.onnx', device='cuda', imgsz=(opt.imgsz, opt.imgsz))
        results['ONNX Runtime CUDA'] = benchmark('ONNX Runtime CUDA', det, img, opt.frames)
    except Exception as e:
        print(f"  ONNX Runtime CUDA        失败: {e}")

    # ONNX Runtime CPU
    try:
        det = YOLOv5OnnxRuntimeDetector('./weights/yolov5s.onnx', device='cpu', imgsz=(opt.imgsz, opt.imgsz))
        results['ONNX Runtime CPU'] = benchmark('ONNX Runtime CPU', det, img, opt.frames)
    except Exception as e:
        print(f"  ONNX Runtime CPU         失败: {e}")

    # TensorRT FP16
    try:
        from tensorrt_detect import YOLOv5TensorRTDetector
        det = YOLOv5TensorRTDetector('./weights/yolov5s.engine', imgsz=(opt.imgsz, opt.imgsz))
        results['TensorRT FP16'] = benchmark('TensorRT FP16', det, img, opt.frames)
    except Exception as e:
        print(f"  TensorRT FP16            失败: {e}")

    # OpenCV DNN CPU
    try:
        from opencv_detect import YOLOv5OpenCVDNN
        det = YOLOv5OpenCVDNN('./weights/yolov5s.onnx', imgsz=(opt.imgsz, opt.imgsz))
        results['OpenCV DNN CPU'] = benchmark('OpenCV DNN CPU', det, img, opt.frames)
    except Exception as e:
        print(f"  OpenCV DNN CPU           失败: {e}")

    print("-" * 55)
    if results:
        fastest = min(results, key=results.get)
        print(f"最快: {fastest} ({results[fastest]:.1f} ms/帧)")


if __name__ == '__main__':
    main()
