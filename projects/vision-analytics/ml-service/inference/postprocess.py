"""Turning raw detector output into final boxes.

A YOLOv8/YOLO11 detection head exported to ONNX returns one tensor of shape
(1, 4 + num_classes, N) where N = 8400 candidate boxes for a 640 input
(80x80 + 40x40 + 20x20 grid cells over the three strides 8, 16, 32).
For each candidate: (cx, cy, w, h) in input pixels, then one sigmoid score per class.

Steps:
1. pick the best class per candidate and drop candidates below `conf_thres`
2. convert centre format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
3. class-wise non-max suppression (NMS) removes duplicate boxes of the same object
4. undo the letterbox so coordinates refer to the original image
"""
from __future__ import annotations

import numpy as np

from .preprocess import LetterboxInfo


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (M, 4) and b (K, 4) in xyxy format -> (M, K).

    IoU = area(intersection) / area(union). 1 means identical boxes, 0 means no overlap.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy NMS. Returns indices of kept boxes, highest score first.

    Repeatedly keep the highest-scoring remaining box and delete every other box that
    overlaps it by more than `iou_thres`.
    """
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thres: float) -> np.ndarray:
    """Class-wise NMS via the offset trick: shift each class's boxes far apart so boxes
    of different classes can never overlap, then run a single NMS."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    offset = classes.astype(np.float32)[:, None] * (boxes.max() + 1.0)
    return nms(boxes + offset, scores, iou_thres)


def scale_boxes(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map xyxy boxes from letterboxed network space back to the original image."""
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - info.pad_x) / info.ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - info.pad_y) / info.ratio
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, info.orig_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, info.orig_h)
    return out


def decode_yolo(output: np.ndarray, info: LetterboxInfo, conf_thres: float = 0.25,
                iou_thres: float = 0.45, max_det: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw (1, 4+C, N) output -> (boxes xyxy (K,4), scores (K,), class ids (K,))."""
    pred = output[0].T                       # (N, 4 + C)
    cls_scores = pred[:, 4:]
    cls_ids = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(pred)), cls_ids]
    mask = scores >= conf_thres
    if not mask.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    boxes = xywh_to_xyxy(pred[mask, :4])
    scores, cls_ids = scores[mask], cls_ids[mask]
    keep = batched_nms(boxes, scores, cls_ids, iou_thres)[:max_det]
    return scale_boxes(boxes[keep], info).astype(np.float32), scores[keep].astype(np.float32), cls_ids[keep]
