"""Smoke test against a RUNNING docker compose stack (server + MongoDB + ML service).

    docker compose up -d --build
    python scripts/docker_smoke.py            # exits non-zero on the first failure

Everything goes through the server on :4100, like the browser: API healthy (MongoDB + ML service up),
a model is loaded, image detection is stored in MongoDB and searchable, a video job runs to completion
in the ML container, and the server serves the React dashboard.
"""
import argparse
import sys
import time

import cv2
import httpx
import numpy as np


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def main(api: str, timeout: float) -> None:
    c = httpx.Client(base_url=api, timeout=30)
    t0 = time.time()
    while True:                                   # the API container needs a few seconds to start
        try:
            h = c.get("/api/health").json()
            if h["status"] == "ok":
                break
            raise RuntimeError(h)
        except Exception:  # noqa: BLE001
            if time.time() - t0 > timeout:
                check(False, f"API reachable at {api}")
            time.sleep(2)
    check(h["mongo"] == "ok" and h["ml"] == "ok" and h["models"],
          f"server healthy (MongoDB + ML service), models: {h['models']}, default: {h['default_model']}")

    img = cv2.imencode(".jpg", np.full((480, 960, 3), 100, np.uint8))[1].tobytes()
    r = c.post("/api/detect/image", files={"file": ("smoke.jpg", img, "image/jpeg")})
    check(r.status_code == 200, f"POST /api/detect/image -> {r.status_code} {r.json().get('counts')}")
    run_id = r.json()["run_id"]
    check(c.get(f"/api/runs/{run_id}").status_code == 200, f"run {run_id} stored in MongoDB")
    check(c.get("/api/runs", params={"q": "smoke"}).json()["total"] >= 1, "history search finds it")

    vw_path = "smoke.mp4"
    vw = cv2.VideoWriter(vw_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 160))
    for i in range(15):
        vw.write(np.full((160, 320, 3), i * 15, np.uint8))
    vw.release()
    with open(vw_path, "rb") as fh:
        r = c.post("/api/detect/video", files={"file": ("smoke.mp4", fh, "video/mp4")})
    check(r.status_code == 202, "POST /api/detect/video accepted (202)")
    vid = r.json()["id"]
    for _ in range(60):
        run = c.get(f"/api/runs/{vid}").json()
        if run["status"] in ("done", "failed"):
            break
        time.sleep(1)
    check(run["status"] == "done", f"video job finished in the container: {run['status']} {run.get('error') or ''}")
    check(c.get(f"/api/runs/{vid}/video").status_code == 200, "playback video served")
    check(c.get("/api/analytics/overview").status_code == 200, "analytics overview (MongoDB aggregation)")

    w = c.get("/history")
    check(w.status_code == 200 and "Vision Analytics" in w.text, "server serves the React dashboard")
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:4100")
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    main(a.api, a.timeout)
