"""A command-line "webcam": streams a video file to the live /api/stream WebSocket exactly like
the browser does (config -> one JPEG frame at a time, waiting for each result -> stop).
Useful for testing the live path on a machine without a camera.

    python scripts/stream_client.py demo/pedestrians.avi --frames 200 --line 0 0.55 1 0.55
"""
import argparse
import asyncio
import json
import time

import cv2
import websockets


async def main(a):
    cap = cv2.VideoCapture(a.video)
    w, h = int(cap.get(3)), int(cap.get(4))
    async with websockets.connect(a.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "config", "model": a.model, "conf": a.conf,
                                  "line": a.line, "width": w, "height": h}))
        ready = json.loads(await ws.recv())
        print("ready:", ready)
        assert ready["type"] == "ready", ready
        t0, n, last = time.perf_counter(), 0, None
        while n < a.frames:
            ok, frame = cap.read()
            if not ok:
                break
            await ws.send(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes())
            last = json.loads(await ws.recv())            # back-pressure: wait for the result
            assert last["type"] == "result", last
            n += 1
        dt = time.perf_counter() - t0
        print(f"{n} frames in {dt:.1f}s -> {n / dt:.1f} FPS round-trip; last frame: "
              f"{len(last['tracks'])} tracks, occupancy {last['occupancy']}, timing {last['timing']}")
        await ws.send(json.dumps({"type": "stop"}))
        saved = json.loads(await ws.recv())
        print("saved:", saved["run_id"], json.dumps(saved["summary"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--url", default="ws://localhost:4100/api/stream")   # the Express relay
    ap.add_argument("--model", default="yolo11n-coco")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--line", type=float, nargs=4, default=None)
    asyncio.run(main(ap.parse_args()))
