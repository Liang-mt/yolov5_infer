# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Segmentation utils (仅保留 plots.py 依赖的 scale_image 函数)
"""

import cv2
import numpy as np


def scale_image(im1_shape, masks, im0_shape, ratio_pad=None):
    """
    将 mask 从模型输入尺寸缩放到原图尺寸。

    Args:
        im1_shape: 模型输入形状 [h, w]
        masks: mask 数组 [h, w, n] 或 [h, w]
        im0_shape: 原图形状 [h, w, c]
        ratio_pad: 可选的 (ratio, pad) 元组

    Returns:
        缩放后的 mask 数组
    """
    # gain = old / new
    if ratio_pad is None:
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    top, left = int(pad[1]), int(pad[0])
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks
