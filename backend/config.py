from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = (PROJECT_ROOT / "storage" / "uploads").resolve()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
