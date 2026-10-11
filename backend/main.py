import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Keep a single import namespace for the backend modules so tests and runtime code
# patch the same router and service modules regardless of how the app is launched.
from api.image_classification import router as image_router
from api.predict import router as predict_router
from api.upload import router as upload_router
from api.train import router as train_router
from api.nlp import router as nlp_router
from api.nlp_routing import router as nlp_routing_router
from config import UPLOAD_DIR
from services.nlp.intent_detection import inspect_model_artifacts
from services.nlp.target_matching import model_dependency_status

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", ",".join(DEFAULT_ALLOWED_ORIGINS)).split(",")
    if origin.strip()
]

app = FastAPI(
    title="TextToAutoML API",
    description="Natural Language Driven Machine Learning Automation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(upload_router)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(nlp_router)
app.include_router(nlp_routing_router)
app.include_router(image_router)


@app.get("/")
def home():
    return {
        "message": "Welcome to TextToAutoML API",
        "status": "Backend Running Successfully",
    }


@app.get("/health")
def health():
    checks = {"uploads_dir": UPLOAD_DIR.is_dir()}
    missing = [name for name, ok in checks.items() if not ok]
    status_value = "ok" if not missing else "degraded"
    return {
        "service": "text-to-automl",
        "status": status_value,
        "checks": checks,
        "missing": missing,
    }


@app.get("/ready")
def readiness():
    intent_artifacts = inspect_model_artifacts()
    checks = {
        "uploads_dir": UPLOAD_DIR.is_dir(),
        "intent_model_artifacts": intent_artifacts["available"],
    }
    optional = {"target_matching_models": model_dependency_status()}
    missing = [name for name, ok in checks.items() if not ok]
    if missing:
        return JSONResponse(
            status_code=503,
            content={
                "service": "text-to-automl",
                "status": "not_ready",
                "checks": checks,
                "missing": missing,
                "artifact_details": intent_artifacts["missing"],
                "optional": optional,
            },
        )
    return {
        "service": "text-to-automl",
        "status": "ok",
        "checks": checks,
        "missing": [],
        "artifact_details": [],
        "optional": optional,
    }