# Local Deployment

Compose builds the backend and the production frontend/Nginx image. The frontend bundle is created inside `Dockerfile.frontend`; a prebuilt `frontend/dist` directory is not required.

## Start

Docker Compose has safe defaults and does not require a `.env` file. To override them, copy `.env.example` to `.env` and edit the local values. The example contains no secrets.

```powershell
docker compose config
docker compose build
docker compose up
```

Open `http://localhost` for the frontend or `http://localhost:8000/health` for the backend health endpoint. The frontend sends same-origin requests through Nginx to the backend. The published backend port can be changed with `BACKEND_PORT`; the internal backend and Nginx port remains `8000`.

## Model Artifacts

The required intent classifier must be available at `INTENT_MODEL_DIR` before `/ready` reports success. The default path is `backend/storage/nlp_models/intent_classifier`; Compose mounts that directory from the host, and the intent training script writes to the same configured path.

Semantic target-matching models are optional for service readiness. Requests that need semantic matching require both configured model names in the host Hugging Face cache at `HF_CACHE_DIR`; Compose mounts that cache read-only into the backend. See [model setup](model-setup.md) for the artifact list and setup commands.

For local Vite development, run the backend separately and start `npm run dev` from `frontend`; set `VITE_API_BASE` to the backend origin. The production frontend does not use the Vite development proxy.