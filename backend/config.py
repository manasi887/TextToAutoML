from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = (PROJECT_ROOT / "storage" / "uploads").resolve()

TARGET_EMBEDDING_MODEL = os.getenv(
	"TARGET_EMBEDDING_MODEL",
	"sentence-transformers/all-MiniLM-L6-v2",
)
TARGET_RERANKER_MODEL = os.getenv(
	"TARGET_RERANKER_MODEL",
	"cross-encoder/ms-marco-MiniLM-L-6-v2",
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
