from fastapi import FastAPI
from api.predict import router as predict_router
from api.upload import router as upload_router
from api.train import router as train_router
from api.nlp import router as nlp_router
from api.cnn import router as cnn_router

app = FastAPI(
    title="TextToAutoML API",
    description="Natural Language Driven Machine Learning Automation",
    version="1.0.0"
)

# Register API routes
app.include_router(upload_router)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(nlp_router)
app.include_router(cnn_router)

@app.get("/")
def home():
    return {
        "message": "Welcome to TextToAutoML API",
        "status": "Backend Running Successfully"
    }