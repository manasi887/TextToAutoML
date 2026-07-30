from fastapi import FastAPI
from api.upload import router as upload_router
from api.train import router as train_router

app = FastAPI(
    title="TextToAutoML API",
    description="Natural Language Driven Machine Learning Automation",
    version="1.0.0"
)

# Register API routes
app.include_router(upload_router)
app.include_router(train_router)


@app.get("/")
def home():
    return {
        "message": "Welcome to TextToAutoML API",
        "status": "Backend Running Successfully"
    }