"""Download small public demo inputs into demo/ (not committed).

  demo/pedestrians.avi - OpenCV's sample clip "vtest.avi" (people walking, fixed camera, ~8 MB):
                         good for testing tracking IDs, line crossings and dwell time.
"""
from pathlib import Path
from urllib.request import urlretrieve

DEMO = Path(__file__).resolve().parents[1] / "demo"
ASSETS = {
    "pedestrians.avi": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
}

if __name__ == "__main__":
    DEMO.mkdir(exist_ok=True)
    for name, url in ASSETS.items():
        dst = DEMO / name
        if not dst.exists():
            print("downloading", url)
            urlretrieve(url, dst)
        print(dst, f"{dst.stat().st_size / 2**20:.1f} MB")
