"""Image preprocessing for a YOLO-style detector.

The network was trained on square 640x640 inputs. Real images are not square, so we
*letterbox* them: resize keeping the aspect ratio, then pad the short side with grey
(114, 114, 114) - the same padding value used during training. Stretching the image
instead would distort object shapes and hurt accuracy.

We return the scale ratio and padding so that boxes predicted in the 640x640 space can
be mapped back to the original image (see postprocess.scale_boxes).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    ratio: float          # resize factor applied to the original image
    pad_x: float          # left padding (pixels, in network-input space)
    pad_y: float          # top padding
    orig_w: int
    orig_h: int


def letterbox(img: np.ndarray, size: int = 640, color: int = 114) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize `img` (H, W, 3) to fit inside size x size, keeping aspect ratio, and pad."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    if (new_w, new_h) != (w, h):
        # INTER_LINEAR = the same interpolation Ultralytics' LetterBox uses, so the network sees
        # pixels just like during validation (verified by the parity check in export_onnx.py)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = size - new_w, size - new_h
    left, top = pad_w // 2, pad_h // 2
    out = cv2.copyMakeBorder(img, top, pad_h - top, left, pad_w - left,
                             cv2.BORDER_CONSTANT, value=(color, color, color))
    return out, LetterboxInfo(ratio=r, pad_x=left, pad_y=top, orig_w=w, orig_h=h)


def to_input_tensor(img_bgr: np.ndarray, size: int = 640) -> tuple[np.ndarray, LetterboxInfo]:
    """BGR uint8 image -> float32 NCHW tensor in [0, 1], plus letterbox info."""
    boxed, info = letterbox(img_bgr, size)
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw[None]), info
