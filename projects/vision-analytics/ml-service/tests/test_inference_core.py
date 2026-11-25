"""Unit tests for the pure-NumPy parts of the pipeline (no model file needed)."""
import numpy as np
import pytest

from inference.analytics import LineCounter, SceneAnalytics, segments_intersect
from inference.kalman import KalmanBoxFilter, xyah_to_xyxy, xyxy_to_xyah
from inference.postprocess import batched_nms, box_iou, decode_yolo, nms, scale_boxes
from inference.preprocess import letterbox, to_input_tensor
from inference.tracker import ByteTracker


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    iou = box_iou(a, b)[0]
    assert iou[0] == pytest.approx(1.0)
    assert iou[1] == pytest.approx(50 / 150)      # intersection 50, union 150
    assert iou[2] == pytest.approx(0.0)


def test_nms_removes_duplicates_keeps_distinct():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]


def test_batched_nms_does_not_suppress_across_classes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    assert sorted(batched_nms(boxes, scores, np.array([0, 1]), 0.5).tolist()) == [0, 1]
    assert batched_nms(boxes, scores, np.array([0, 0]), 0.5).tolist() == [0]


def test_letterbox_shape_and_roundtrip():
    img = np.zeros((480, 1280, 3), np.uint8)
    out, info = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert info.ratio == pytest.approx(0.5)
    assert info.pad_x == 0 and info.pad_y == 200       # 480*0.5 = 240 px tall; (640-240)/2 = 200
    # a box drawn in network space maps back to original-image space
    back = scale_boxes(np.array([[0, 200, 640, 440]], np.float32), info)
    np.testing.assert_allclose(back[0], [0, 0, 1280, 480])
    x, _ = to_input_tensor(img, 640)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32


def test_decode_yolo_synthetic_output():
    """Build a fake (1, 4+C, N) head output and check decode + NMS + rescale."""
    C, N = 3, 5
    out = np.zeros((1, 4 + C, N), np.float32)
    # candidate 0 and 1: same object (class 2), candidate 2: different object (class 0)
    out[0, :4, 0] = [100, 100, 40, 40]; out[0, 4 + 2, 0] = 0.9
    out[0, :4, 1] = [102, 101, 40, 40]; out[0, 4 + 2, 1] = 0.6
    out[0, :4, 2] = [400, 300, 80, 60]; out[0, 4 + 0, 2] = 0.8
    out[0, 4 + 1, 3] = 0.1                           # below threshold
    _, info = letterbox(np.zeros((640, 640, 3), np.uint8))
    boxes, scores, cls = decode_yolo(out, info, conf_thres=0.25, iou_thres=0.45)
    assert cls.tolist() == [2, 0]
    np.testing.assert_allclose(boxes[0], [80, 80, 120, 120])
    np.testing.assert_allclose(scores, [0.9, 0.8], rtol=1e-6)


def test_kalman_tracks_constant_velocity():
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate(xyxy_to_xyah(np.array([0, 0, 10, 20.0])))
    for step in range(1, 15):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, xyxy_to_xyah(np.array([5 * step, 0, 5 * step + 10, 20.0])))
    pred, _ = kf.predict(mean, cov)
    assert pred[4] == pytest.approx(5.0, abs=0.3)        # learned velocity of 5 px/frame
    np.testing.assert_allclose(xyah_to_xyxy(pred), [75, 0, 85, 20], atol=1.0)


def _moving_box(x, y):
    return [x, y, x + 40, y + 80]


def test_tracker_keeps_ids_and_survives_short_occlusion():
    trk = ByteTracker()
    ids_a, ids_b = set(), set()
    for f in range(40):
        boxes, scores = [_moving_box(10 + 4 * f, 50)], [0.9]
        if not 15 <= f < 20:                              # object B disappears for 5 frames
            boxes.append(_moving_box(400 - 4 * f, 200)); scores.append(0.85)
        tracks = trk.update(np.array(boxes, float), np.array(scores), np.zeros(len(boxes), int))
        for t in tracks:
            (ids_a if t.centre[1] < 150 else ids_b).add(t.track_id)
    assert len(ids_a) == 1, ids_a
    assert len(ids_b) == 1, ids_b                         # same ID after the occlusion
    assert ids_a != ids_b


def test_tracker_low_score_detection_keeps_track_alive():
    trk = ByteTracker()
    for f in range(5):
        trk.update(np.array([_moving_box(10 + 3 * f, 50)], float), np.array([0.9]), np.array([0]))
    # partially occluded: the detector is unsure (0.3) but round-2 matching still links it
    tracks = trk.update(np.array([_moving_box(25, 50)], float), np.array([0.3]), np.array([0]))
    assert [t.track_id for t in tracks] == [1]


def test_tracker_does_not_confirm_single_frame_noise():
    trk = ByteTracker()
    trk.update(np.array([_moving_box(10, 10)], float), np.array([0.9]), np.array([0]))  # frame 1 seeds track 1
    out = trk.update(np.array([_moving_box(10, 10), _moving_box(300, 300)], float),
                     np.array([0.9, 0.9]), np.array([0, 0]))
    assert [t.track_id for t in out] == [1]               # new object needs a 2nd frame


def test_segments_intersect():
    assert segments_intersect((0, -1), (0, 1), (-5, 0), (5, 0))
    assert not segments_intersect((10, -1), (10, 1), (-5, 0), (5, 0))   # crosses the extension only


def test_line_counter_counts_each_track_once_per_direction():
    trk = ByteTracker()
    scene = SceneAnalytics(fps=10, line=LineCounter((0, 200), (1000, 200)))
    for f in range(30):                                   # walks downwards across y = 200
        tracks = trk.update(np.array([[100, 100 + 10 * f, 140, 180 + 10 * f]], float),
                            np.array([0.9]), np.array([0]))
        scene.update(tracks, {0: "person"})
    s = scene.summary()
    # A=(0,200) -> B=(1000,200): points above the line (y < 200) are on the negative side,
    # so moving downwards is negative -> positive = "in"
    assert s["line"] == {"in": {"person": 1}, "out": {}}
    assert s["unique_objects"] == {"person": 1}
    assert s["avg_dwell_seconds"]["person"] == pytest.approx(3.0)
