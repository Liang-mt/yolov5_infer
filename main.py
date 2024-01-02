from main_detect import loadmodel, detect, detectdraw
import cv2
import torch

if __name__ == '__main__':
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = './weights/yolov5s.pt'  # 权重文件地址   .pt文件
    model, imgsz = loadmodel(weights, device)

    frame = cv2.imread("./images/bus.jpg")
    result = detect(frame, device, model, imgsz)

    # 绘制识别框
    for detection in result:
        detectdraw(frame, detection)

    # 在窗口中显示帧
    cv2.imshow('Video', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()