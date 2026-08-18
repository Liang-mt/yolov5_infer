# YOLOv5 多后端检测器基类
import numpy as np
import cv2


class YOLOv5BaseDetector:
    """封装公共的预处理、后处理和绘制逻辑，各后端子类只需实现模型加载和推理。"""

    # COCO 80 类名称
    COCO_NAMES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
        34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
        38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
        43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
        48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
        53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
        58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
        63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
        68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
        73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
        78: 'hair drier', 79: 'toothbrush',
    }

    def __init__(self, imgsz, conf_thres, iou_thres, max_det,
                 classes, agnostic_nms, names=None):
        # imgsz 校验：自动对齐到 32 的倍数（YOLOv5 最大 stride）
        if isinstance(imgsz, int):
            imgsz = (imgsz, imgsz)
        h = max(32, (imgsz[0] + 31) // 32 * 32)
        w = max(32, (imgsz[1] + 31) // 32 * 32)
        if (h, w) != tuple(imgsz):
            print(f"[警告] imgsz={imgsz} 不是 32 的倍数，已自动调整为 {(h, w)}")
        self.imgsz = (h, w)

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.names = names if names is not None else self.COCO_NAMES

    # ---- 预处理 ----

    @staticmethod
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
        """缩放并填充图片到指定尺寸，保持宽高比。"""
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    # ---- 后处理 ----

    @staticmethod
    def scale_boxes(xyxy, img1_shape, img0_shape):
        """将检测框从模型输入尺寸缩放到原图尺寸。"""
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)
        xyxy[:, [0, 2]] -= pad[0]
        xyxy[:, [1, 3]] -= pad[1]
        xyxy[:, :4] /= gain
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, img0_shape[1])
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, img0_shape[0])
        return xyxy

    @staticmethod
    def nms(boxes, scores, iou_thres):
        """纯 numpy 实现的非极大值抑制。"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=np.int64)

    def postprocess(self, pred, im0_shape, conf_thres=None, iou_thres=None,
                    max_det=None, classes=None, agnostic_nms=None):
        """通用后处理：从 YOLOv5 原始输出提取检测结果。

        Args:
            pred: numpy array, shape [N, 5+nc]，每行 [cx,cy,w,h,obj_conf,class_conf...]
            im0_shape: 原图尺寸 (h, w)
            其余参数不传则用实例默认值

        Returns:
            list[dict]: [{'class': str, 'conf': float, 'position': [l,t,w,h]}, ...]
        """
        conf_thres = conf_thres if conf_thres is not None else self.conf_thres
        iou_thres = iou_thres if iou_thres is not None else self.iou_thres
        max_det = max_det if max_det is not None else self.max_det
        classes = classes if classes is not None else self.classes
        agnostic_nms = agnostic_nms if agnostic_nms is not None else self.agnostic_nms

        obj_conf = pred[:, 4]
        class_conf = pred[:, 5:]

        mask = obj_conf > conf_thres
        pred = pred[mask]
        if len(pred) == 0:
            return []

        obj_conf = obj_conf[mask]
        class_conf = class_conf[mask]
        cls_id = class_conf.argmax(axis=1)
        cls_conf = class_conf[np.arange(len(pred)), cls_id]
        scores = obj_conf * cls_conf

        # xywh(中心) -> xyxy
        boxes = np.empty((len(pred), 4), dtype=np.float32)
        boxes[:, 0] = pred[:, 0] - pred[:, 2] / 2
        boxes[:, 1] = pred[:, 1] - pred[:, 3] / 2
        boxes[:, 2] = pred[:, 0] + pred[:, 2] / 2
        boxes[:, 3] = pred[:, 1] + pred[:, 3] / 2

        if classes is not None:
            cls_mask = np.isin(cls_id, classes)
            boxes, scores, cls_id = boxes[cls_mask], scores[cls_mask], cls_id[cls_mask]
            if len(boxes) == 0:
                return []

        if agnostic_nms:
            keep = self.nms(boxes, scores, iou_thres)
        else:
            keep = []
            for c in np.unique(cls_id):
                idx = np.where(cls_id == c)[0]
                k = self.nms(boxes[idx], scores[idx], iou_thres)
                keep.extend(idx[k].tolist())
            keep = np.array(keep, dtype=np.int64)

        if len(keep) == 0:
            return []
        if len(keep) > max_det:
            keep = keep[np.argsort(scores[keep])[::-1][:max_det]]

        boxes = self.scale_boxes(boxes[keep], self.imgsz, im0_shape)

        detections = []
        for box, score, cid in zip(boxes, scores[keep], cls_id[keep]):
            x1, y1, x2, y2 = [round(v) for v in box]
            detections.append({
                'class': self.names.get(int(cid), f'class{int(cid)}'),
                'conf': round(float(score), 2),
                'position': [x1, y1, x2 - x1, y2 - y1],
            })
        return detections

    # ---- 绘制 ----

    @staticmethod
    def draw(frame, detection, color=(0, 255, 0), line_thickness=2):
        """在图片上绘制单个检测框和标签。"""
        cls = detection['class']
        conf = detection['conf']
        x, y, w, h = detection['position']
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, line_thickness)
        caption = "{} {:.2f}".format(cls, conf)
        # 标签位置：默认在框上方，贴顶时画到框内顶部
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_thickness)
        label_y = y - 8 if y - th - 8 >= 0 else y + th + 4
        cv2.putText(frame, caption, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,
                    thickness=line_thickness, lineType=cv2.LINE_AA)

    def __call__(self, img, **kwargs):
        return self.detect(img, **kwargs)
