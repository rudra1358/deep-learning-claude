"""ML service API tests against the fake ONNX detector (see conftest.py).

Storage (runs, history, analytics) is the Express server's job and is tested in server/tests.
Here we test what only Python does: detection, comparison, video jobs and the live WebSocket.
"""
import json
import shutil
import time
import uuid

import cv2
import numpy as np
import pytest


def _jpeg(w=1280, h=640) -> bytes:
    img = np.full((h, w, 3), 90, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _video(path, frames=20, w=320, h=160) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(frames):
        vw.write(np.full((h, w, 3), i * 10 % 255, np.uint8))
    vw.release()


def _upload(client, src=None, data: bytes | None = None, ext=".mp4"):
    """Put a file into data/uploads the way the Express server does; return its path."""
    dest = client.data_dir / "uploads" / f"{uuid.uuid4().hex}{ext}"
    if src is not None:
        shutil.copy(src, dest)
    else:
        dest.write_bytes(data or b"")
    return str(dest)


def test_health_and_models(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["models"] == ["fake"]
    models = client.get("/models").json()
    assert models[0]["name"] == "fake" and models[0]["loaded"] is True
    assert models[0]["classes"] == {"0": "person", "1": "car"}


def test_detect_image_decodes_nms_and_rescales(client):
    r = client.post("/detect/image", files={"file": ("a.jpg", _jpeg(), "image/jpeg")},
                    data={"conf": "0.25", "iou": "0.45"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == 1280 and body["height"] == 640
    assert body["counts"] == {"person": 1, "car": 1}          # the duplicate person was suppressed
    person = next(d for d in body["detections"] if d["label"] == "person")
    # network box (128,256)-(192,384); letterbox ratio 0.5, pad_y 160 -> original coordinates
    assert person["box"] == pytest.approx([256, 192, 384, 448], abs=0.2)
    assert set(body["timing"]) == {"preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"}


def test_confidence_threshold_filters(client):
    r = client.post("/detect/image", files={"file": ("a.jpg", _jpeg(), "image/jpeg")},
                    data={"conf": "0.85"}).json()
    assert r["counts"] == {"person": 1}


@pytest.mark.parametrize("files,data,code", [
    ({"file": ("a.txt", b"hello", "text/plain")}, {}, 415),
    ({"file": ("a.jpg", b"not an image", "image/jpeg")}, {}, 400),
    ({"file": ("a.jpg", _jpeg(), "image/jpeg")}, {"conf": "1.5"}, 422),
    ({"file": ("a.jpg", _jpeg(), "image/jpeg")}, {"model": "nope"}, 404),
])
def test_detect_image_validation(client, files, data, code):
    assert client.post("/detect/image", files=files, data=data).status_code == code


def test_compare(client):
    r = client.post("/compare", files={"file": ("a.jpg", _jpeg(), "image/jpeg")},
                    data={"models": "fake,fake"}).json()
    assert len(r["results"]) == 2 and r["results"][0]["counts"] == r["results"][1]["counts"]


def _wait_done(client, run_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/jobs/{run_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.2)
    raise TimeoutError


def test_video_job_end_to_end(client, tmp_path):
    p = tmp_path / "clip.mp4"
    _video(p)
    r = client.post("/jobs/video", json={"run_id": "run1", "video_path": _upload(client, src=p),
                                         "line": [0, 0.5, 1, 0.5]})
    assert r.status_code == 202, r.text
    job = _wait_done(client, "run1")
    assert job["status"] == "done", job["error"]
    assert job["fps"] == pytest.approx(10) and job["width"] == 320
    res = job["result"]
    assert res["frames"] == 20 and res["avg_latency_ms"] > 0
    assert res["summary"]["unique_objects"] == {"person": 1, "car": 1}   # 2 stable track IDs
    assert res["summary"]["peak_occupancy"] == {"person": 1, "car": 1}
    assert sorted(t["track_id"] for t in res["tracks"]) == [1, 2]
    assert {s["t_seconds"] for s in res["samples"]} == {0.0, 1.0}

    run_dir = client.data_dir / "runs" / "run1"
    frames = json.loads((run_dir / "frames.json").read_text())
    assert len(frames["frames"]) == 20
    assert {t["id"] for f in frames["frames"] for t in f["tracks"]} == {1, 2}
    cap = cv2.VideoCapture(str(run_dir / "playback.webm"))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == 20                                            # playback copy has exactly the analysed frames

    assert client.delete("/jobs/run1").status_code == 204
    assert client.get("/jobs/run1").status_code == 404


def test_video_job_validation(client, tmp_path):
    ok_path = _upload(client, data=b"00")
    assert client.post("/jobs/video", json={"run_id": "v1", "video_path": ok_path, "line": [0, 2, 1, 1]}
                       ).status_code == 422
    assert client.post("/jobs/video", json={"run_id": "v2", "video_path": ok_path, "model": "nope"}
                       ).status_code == 404
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"00")                                # not in uploads/ -> refused
    assert client.post("/jobs/video", json={"run_id": "v3", "video_path": str(outside)}).status_code == 400
    assert client.post("/jobs/video", json={"run_id": "../x", "video_path": ok_path}).status_code == 422


def test_broken_video_marks_job_failed(client):
    client.post("/jobs/video", json={"run_id": "bad1", "video_path": _upload(client, data=b"garbage" * 100)})
    job = _wait_done(client, "bad1")
    assert job["status"] == "failed" and job["error"]


def test_websocket_stream_returns_session(client):
    with client.websocket_connect("/stream") as ws:
        ws.send_json({"type": "config", "model": "fake", "conf": 0.3, "width": 1280, "height": 640,
                      "line": None})
        assert ws.receive_json()["type"] == "ready"
        for _ in range(3):
            ws.send_bytes(_jpeg())
            msg = ws.receive_json()
            assert msg["type"] == "result"
        assert msg["occupancy"] == {"person": 1, "car": 1}
        assert sorted(t["id"] for t in msg["tracks"]) == [1, 2]       # IDs persist across frames
        ws.send_json({"type": "stop"})
        session = ws.receive_json()
    assert session["type"] == "session"
    run = session["run"]
    assert run["kind"] == "stream" and run["frames"] == 3 and len(run["tracks"]) == 2


def test_websocket_requires_config(client):
    with client.websocket_connect("/stream") as ws:
        ws.send_bytes(_jpeg())
        assert ws.receive_json()["type"] == "error"


def test_websocket_validates_config_and_answers_empty_stop(client):
    with client.websocket_connect("/stream") as ws:
        ws.send_json({"type": "config", "model": "fake", "line": [0, 2, 1]})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "config", "model": "fake", "conf": 5})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "config", "model": "fake"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "stop"})
        assert ws.receive_json() == {"type": "session", "run": None}
