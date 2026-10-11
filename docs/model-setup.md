# Model setup on a clean machine

TextToAutoML uses a required local intent classifier and optional local semantic target-matching models:

1. Intent-classification model for user-message routing
   - Default directory: `backend/storage/nlp_models/intent_classifier`
   - Required files: tokenizer, config, model weights, and `label_mapping.json`
   - Serving, readiness, and the training script use the same `INTENT_MODEL_DIR` setting.
   - Readiness checks for `config.json`, tokenizer files, model weights, and `label_mapping.json`.

2. Semantic target-matching models for column matching
   - Default Hugging Face names:
     - `sentence-transformers/all-MiniLM-L6-v2`
     - `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - These are loaded with `local_files_only=True`, so requests needing semantic matching require them in the Hugging Face cache.
   - They are optional for application readiness; affected NLP requests return 503 if unavailable.

## Quick setup

```powershell
Copy-Item .env.example .env
New-Item -ItemType Directory -Force backend/storage/nlp_models/intent_classifier | Out-Null
New-Item -ItemType Directory -Force .cache/huggingface | Out-Null
```

Download the semantic models into the configured Hugging Face cache:

```powershell
$env:HF_HOME = (Resolve-Path .cache/huggingface).Path
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2'); AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

Train the intent classifier with `python -m services.nlp.train_intent_model` from the `backend` directory. Its output is written to the configured `INTENT_MODEL_DIR` (default `backend/storage/nlp_models/intent_classifier`).

For Compose, the frontend image builds its Vite bundle as part of `docker compose up --build`; there is no pre-existing `frontend/dist` requirement. Start the stack from the repository root with `docker compose up --build`. `/ready` remains 503 until the intent classifier artifacts are present.
