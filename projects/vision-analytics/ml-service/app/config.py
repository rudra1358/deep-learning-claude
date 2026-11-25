"""Application settings, read from environment variables (12-factor style).

Every value can be overridden with an env var of the same name prefixed with VA_,
e.g. VA_DEVICE=cpu. There is no database setting: MongoDB belongs to the Express server.
VA_DATA_DIR must be the same folder the Express server uses (it saves uploads there and
this service writes frames.json / playback.webm next to them).
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]          # projects/vision-analytics


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VA_", env_file=".env", extra="ignore")

    models_dir: Path = ROOT / "models"
    data_dir: Path = ROOT / "data"
    metrics_file: Path = ROOT / "training" / "reports" / "metrics.json"
    device: str = "auto"                 # auto | cuda | cpu
    default_model: str = "yolo11n-coco"     # general scenes / webcam; *-visdrone for aerial traffic
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_upload_mb: int = 200
    max_video_seconds: int = 600


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    (s.data_dir / "uploads").mkdir(exist_ok=True)
    (s.data_dir / "runs").mkdir(exist_ok=True)
    return s
