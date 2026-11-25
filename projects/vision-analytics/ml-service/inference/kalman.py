"""Constant-velocity Kalman filter for bounding boxes (the SORT / ByteTrack formulation).

State x (8-dim): [cx, cy, a, h, vcx, vcy, va, vh]
  cx, cy = box centre, a = aspect ratio w/h, h = height, v* = their velocities.
Measurement z (4-dim): [cx, cy, a, h] - what the detector gives us each frame.

Two steps every frame:
  predict:  x' = F x,         P' = F P F^T + Q      (move the box by its velocity)
  update:   K  = P' H^T (H P' H^T + R)^-1           (Kalman gain: how much to trust the detector)
            x  = x' + K (z - H x'),  P = (I - K H) P'
Process noise Q and measurement noise R scale with box height, so big (near) objects
are allowed to move more pixels per frame than small (far) ones.
"""
from __future__ import annotations

import numpy as np

_STD_POS = 1.0 / 20
_STD_VEL = 1.0 / 160


class KalmanBoxFilter:
    ndim = 4

    def __init__(self) -> None:
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0          # position += velocity (dt = 1 frame)
        self.H = np.eye(4, 8)               # we observe the first 4 state entries

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * _STD_POS * h, 2 * _STD_POS * h, 1e-2, 2 * _STD_POS * h,
               10 * _STD_VEL * h, 10 * _STD_VEL * h, 1e-5, 10 * _STD_VEL * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [_STD_POS * h, _STD_POS * h, 1e-2, _STD_POS * h,
               _STD_VEL * h, _STD_VEL * h, 1e-5, _STD_VEL * h]
        Q = np.diag(np.square(std))
        return self.F @ mean, self.F @ cov @ self.F.T + Q

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        R = np.diag(np.square([_STD_POS * h, _STD_POS * h, 1e-1, _STD_POS * h]))
        S = self.H @ cov @ self.H.T + R                 # innovation covariance
        K = np.linalg.solve(S, self.H @ cov).T          # = cov H^T S^-1 (solve is more stable than inv)
        new_mean = mean + K @ (z - self.H @ mean)
        new_cov = cov - K @ S @ K.T
        return new_mean, new_cov


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(s: np.ndarray) -> np.ndarray:
    cx, cy, a, h = s[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)
