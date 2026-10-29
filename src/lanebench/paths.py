"""데이터 · 결과 경로. 환경 변수로 덮어쓸 수 있습니다."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# BDD100K v1 per-image 라벨 JSON (labels/100k/{train,val}/<name>.json)
LABEL_DIR = Path(os.environ.get(
    "LANEBENCH_LABELS",
    Path.home() / "data/bdd100k_yolop/data2/zwt/bdd/bdd100k/labels/100k"))

# BDD100K 100k 이미지 (images/100k/{train,val}/<name>.jpg, 하위 폴더가 있어도 됨)
IMAGE_DIR = Path(os.environ.get(
    "LANEBENCH_IMAGES",
    Path.home() / "data/bdd100k/images/100k"))

RESULTS_DIR = Path(os.environ.get("LANEBENCH_RESULTS", REPO_ROOT / "results"))
RAW_DIR = RESULTS_DIR / "raw"
CKPT_DIR = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"
FIG_DIR = RESULTS_DIR / "figures"

MANIFEST = RAW_DIR / "manifest.csv"
