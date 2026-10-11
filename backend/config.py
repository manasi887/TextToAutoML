import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = (PROJECT_ROOT / "storage" / "uploads").resolve()
INTENT_MODEL_DIR = Path(
    os.getenv(
        "INTENT_MODEL_DIR",
        BACKEND_DIR / "storage" / "nlp_models" / "intent_classifier",
    )
).resolve()

TARGET_EMBEDDING_MODEL = os.getenv(
    "TARGET_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
TARGET_RERANKER_MODEL = os.getenv(
    "TARGET_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
ALLOWED_ORIGINS = [
    value.strip()
    for value in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if value.strip()
]

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
