"""End-to-end API tests against the fake ONNX detector (see conftest.py)."""
import json
import time

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


def test_health_and_models(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["models"] == ["fake"]
    models = client.get("/api/models").json()
    assert models[0]["name"] == "fake" and models[0]["loaded"] is True
    assert models[0]["classes"] == {"0": "person", "1": "car"}


def test_detect_image_decodes_nms_and_rescales(client):
    r = client.post("/api/detect/image", files={"file": ("a.jpg", _jpeg(), "image/jpeg")},
                    data={"conf": "0.25", "iou": "0.45"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == 1280 and body["height"] == 640
    assert body["counts"] == {"person": 1, "car": 1}          # the duplicate person was suppressed
    person = next(d for d in body["detections"] if d["label"] == "person")
    # network box (128,256)-(192,384); letterbox ratio 0.5, pad_y 160 -> original coordinates
    assert person["box"] == pytest.approx([256, 192, 384, 448], abs=0.2)
    assert body["run_id"] is not None
    assert set(body["timing"]) == {"preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"}


def test_confidence_threshold_filters(client):
    r = client.post("/api/detect/image", files={"file": ("a.jpg", _jpeg(), "image/jpeg")},
                    data={"conf": "0.85", "save": "false"}).json()
    assert r["counts"] == {"person": 1} and r["run_id"] is None


@pytest.mark.parametrize("files,data,code", [
    ({"file": ("a.txt", b"hello", "text/plain")}, {}, 415),
    ({"file": ("a.jpg", b"not an image", "image/jpeg")}, {}, 400),
    ({"file": ("a.jpg", _jpeg(), "image/jpeg")}, {"conf": "1.5"}, 422),
    ({"file": ("a.jpg", _jpeg(), "image/jpeg")}, {"model": "nope"}, 404),
])
def test_detect_image_validation(client, files, data, code):
    assert client.post("/api/detect/image", files=files, data=data).status_code == code


def test_compare(client):
    r = client.post("/api/compare", files={"file": ("a.jpg", _jpeg(), "image/jpeg")},
                    data={"models": "fake,fake"}).json()
    assert len(r["results"]) == 2 and r["results"][0]["counts"] == r["results"][1]["counts"]


def _wait_done(client, run_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in ("done", "failed"):
            return run
        time.sleep(0.2)
    raise TimeoutError


def test_video_job_end_to_end(client, tmp_path):
    p = tmp_path / "clip.mp4"
    _video(p)
    with p.open("rb") as fh:
        r = client.post("/api/detect/video", files={"file": ("clip.mp4", fh, "video/mp4")},
                        data={"line": json.dumps([0, 0.5, 1, 0.5])})
    assert r.status_code == 202, r.text
    run = _wait_done(client, r.json()["id"])
    assert run["status"] == "done", run["error"]
    assert run["frames"] == 20 and run["fps"] == pytest.approx(10)
    assert run["summary"]["unique_objects"] == {"person": 1, "car": 1}   # 2 stable track IDs
    assert run["summary"]["peak_occupancy"] == {"person": 1, "car": 1}
    frames = client.get(f"/api/runs/{run['id']}/frames").json()
    assert len(frames["frames"]) == 20
    assert {t["id"] for f in frames["frames"] for t in f["tracks"]} == {1, 2}
    ts = client.get(f"/api/runs/{run['id']}/timeseries").json()
    assert ts["labels"] == ["car", "person"] and ts["t"] == [0.0, 1.0]
    csv = client.get(f"/api/runs/{run['id']}/export.csv").text.splitlines()
    assert csv[0].startswith("track_id,label") and len(csv) == 3
    video = client.get(f"/api/runs/{run['id']}/video")
    assert video.status_code == 200 and video.headers["content-type"] == "video/webm"
    (tmp_path / "pb.webm").write_bytes(video.content)
    cap = cv2.VideoCapture(str(tmp_path / "pb.webm"))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == 20                                            # playback copy has exactly the analysed frames


def test_video_validation(client):
    assert client.post("/api/detect/video", files={"file": ("x.exe", b"MZ", "application/octet-stream")}
                       ).status_code == 415
    assert client.post("/api/detect/video", files={"file": ("x.mp4", b"00", "video/mp4")},
                       data={"line": "[0, 2, 1]"}).status_code == 422


def test_broken_video_marks_run_failed(client):
    r = client.post("/api/detect/video", files={"file": ("bad.mp4", b"garbage" * 100, "video/mp4")})
    run = _wait_done(client, r.json()["id"])
    assert run["status"] == "failed" and run["error"]


def test_websocket_stream_and_save(client):
    with client.websocket_connect("/api/stream") as ws:
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
        saved = ws.receive_json()
        assert saved["type"] == "saved"
    run = client.get(f"/api/runs/{saved['run_id']}").json()
    assert run["kind"] == "stream" and run["frames"] == 3


def test_websocket_requires_config(client):
    with client.websocket_connect("/api/stream") as ws:
        ws.send_bytes(_jpeg())
        assert ws.receive_json()["type"] == "error"


def test_websocket_validates_config_and_answers_empty_stop(client):
    with client.websocket_connect("/api/stream") as ws:
        ws.send_json({"type": "config", "model": "fake", "line": [0, 2, 1]})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "config", "model": "fake", "conf": 5})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "config", "model": "fake"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "stop"})
        assert ws.receive_json() == {"type": "saved", "run_id": None, "summary": None}


def test_created_at_is_timezone_aware(client):
    client.post("/api/detect/image", files={"file": ("tz.jpg", _jpeg(), "image/jpeg")})
    created = client.get("/api/runs").json()["items"][0]["created_at"]
    assert created.endswith("Z") or created.endswith("+00:00")


def test_history_filters_overview_and_delete(client):
    client.post("/api/detect/image", files={"file": ("street.jpg", _jpeg(), "image/jpeg")})
    page = client.get("/api/runs", params={"q": "street"}).json()
    assert page["total"] >= 1 and all("street" in r["source_name"] for r in page["items"])
    assert client.get("/api/runs", params={"label": "car"}).json()["total"] >= 1
    assert client.get("/api/runs", params={"label": "giraffe"}).json()["total"] == 0
    assert client.get("/api/runs", params={"kind": "stream"}).json()["total"] >= 1
    ov = client.get("/api/analytics/overview").json()
    assert ov["objects_by_label"]["person"] >= 1 and ov["avg_latency_ms"] > 0
    rid = page["items"][0]["id"]
    assert client.delete(f"/api/runs/{rid}").status_code == 204
    assert client.get(f"/api/runs/{rid}").status_code == 404
