# TextToAutoML

TextToAutoML is a natural-language-driven machine learning application for tabular datasets and image classification. Users can upload data, describe a task, review the inferred intent and target, train a model, and use the saved model to make predictions.

The project combines a React frontend, a FastAPI backend, dataset intelligence, NLP routing, and task-specific training pipelines. It is an academic final-year project; model selection and reported metrics should be evaluated in the context of each dataset and run.

## Project status

| Capability | Status |
| --- | --- |
| Tabular CSV/XLSX upload, validation, and analysis | Implemented |
| Tabular preprocessing and dataset recommendations | Implemented |
| NLP intent detection, clarification, and target resolution | Implemented |
| Classification, regression, and clustering workflows | Implemented |
| Tabular model persistence and Model ID prediction | Implemented |
| Training reports and factual result explanations | Implemented |
| Image ZIP analysis, classification training, and prediction | Implemented |
| React frontend and Docker Compose deployment | Implemented |
| BART-based explanation layer | Planned; not implemented |

## Workflows

### Tabular data

1. Upload a CSV or XLSX dataset.
2. Inspect validation, analysis, preprocessing, and dataset-intelligence results.
3. Enter a natural-language task; the NLP pipeline detects intent and, for supervised tasks, extracts and matches a target column.
4. Review the task resolution and training estimate. Ambiguous intent or target matches require clarification.
5. Train through the NLP workflow or explicitly select the target and train through the direct AutoML endpoint.
6. Review evaluation results, the saved Model ID, the training report, and the result explanation.
7. Predict from new feature records or a CSV/XLSX file using the saved Model ID.

Clustering does not require a supervised target column. Its pipeline prepares mixed-type inputs, evaluates prediction-compatible clusterers, and saves the selected fitted pipeline for assignment of new records.

### Image classification

The image workflow accepts a ZIP image dataset with class folders, analyzes its structure, trains a MobileNetV2 classifier, saves a model package, and predicts the class of an uploaded image with a saved image Model ID. The image workflow is separate from tabular AutoML.

## Architecture

```text
React + Vite frontend
        ↓
FastAPI API
  ├── Dataset upload and validation
  ├── NLP analysis, routing, and training
  ├── Tabular AutoML and prediction
  └── Image dataset analysis, training, and prediction
        ↓
Dataset / NLP / AutoML / image services
        ↓
Persistent uploaded data and model packages
```

## Repository layout

```text
TextToAutoML/
├── backend/
│   ├── api/                 # FastAPI routers
│   ├── services/
│   │   ├── automl/          # Training, evaluation, persistence, prediction
│   │   ├── dataset/         # Loading, validation, analysis, preprocessing
│   │   ├── explanation/     # AutoML result explanation
│   │   ├── image_classification/
│   │   ├── metalearning/
│   │   ├── nlp/
│   │   └── reporting/
│   ├── config.py
│   └── main.py
├── frontend/                # React app and frontend tests
├── tests/                   # Python backend tests
├── docs/                    # Deployment and model setup guides
├── storage/                 # Persistent uploaded data and model packages
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.frontend
└── .env.example
```

## Technology stack

- **Frontend:** React 19, Vite 7, JavaScript
- **API:** FastAPI, Pydantic, Uvicorn
- **Data and tabular ML:** pandas, NumPy, scikit-learn
- **NLP and image models:** PyTorch, Hugging Face Transformers, torchvision
- **Persistence:** joblib-based tabular packages and separate image-model packages
- **Deployment:** Docker, Docker Compose, Nginx

## Quick start

### Prerequisites

- Python environment compatible with the pinned packages in `requirements.txt` (the backend Docker image uses Python 3.12)
- Node.js and npm for local frontend development
- Docker Compose if using the containerized setup

### 1. Clone and install Python dependencies

```powershell
git clone https://github.com/manasi887/TextToAutoML.git
cd TextToAutoML
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Prepare NLP model artifacts

NLP intent detection requires a trained classifier in `INTENT_MODEL_DIR`. The default is `backend/storage/nlp_models/intent_classifier`. The model setup guide explains how to train it and what files readiness checks require:

- [Model setup](docs/model-setup.md)

Semantic target matching uses local Hugging Face models when needed. They are optional for application readiness; requests that require semantic matching need the model files in the configured Hugging Face cache. See the model setup guide for details.

### 3. Run the backend

From the repository root, with the virtual environment active:

```powershell
python -m uvicorn main:app --app-dir backend --reload --port 8000
```

The API is available at `http://127.0.0.1:8000`; interactive documentation is at `http://127.0.0.1:8000/docs`.

### 4. Run the frontend

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Vite runs on port `5173` and proxies API paths to `http://127.0.0.1:8000` by default. Set `VITE_API_BASE` to use a different backend origin.

## Docker deployment

Compose builds the backend and production frontend. The frontend bundle is built in the image; a local `frontend/dist` directory is not required. To use the documented defaults:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open `http://localhost` for the frontend or `http://localhost:8000/health` for backend health. The backend's published host port can be changed with `BACKEND_PORT`. Configuration options and artifact mounts are described in [Local deployment](docs/deployment.md).

## API overview

The routes below are registered by the FastAPI application. Request schemas and response details are available at `/docs`.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/upload/` | Upload and inspect a tabular CSV or XLSX dataset |
| `POST` | `/nlp/analyze` | Analyze a natural-language task against an uploaded dataset |
| `POST` | `/nlp/route` | Route a request using its text and supplied asset context |
| `POST` | `/nlp/train` | Resolve an NLP request and train when ready |
| `POST` | `/nlp/diagnose-target-proxies` | Diagnose possible target-proxy features |
| `POST` | `/train/` | Train a tabular model with a specified target column |
| `POST` | `/predict/` | Predict from feature records and a saved tabular Model ID |
| `POST` | `/predict/file` | Predict from a CSV/XLSX file and a saved tabular Model ID |
| `POST` | `/image/analyze` | Analyze an uploaded ZIP image dataset |
| `POST` | `/image/train` | Train an image classifier from an uploaded ZIP dataset |
| `POST` | `/image/predict` | Classify one uploaded image using a saved image Model ID |
| `GET` | `/health` | Check basic backend health |
| `GET` | `/ready` | Check required intent-classifier artifacts and readiness |

## NLP intent and target resolution

Intent detection uses a locally stored fine-tuned DistilBERT sequence classifier. The classifier distinguishes `classification`, `regression`, `clustering`, and `unknown`. A confidence-and-margin check can request clarification rather than proceed with an uncertain or unknown intent. Explicit task language may resolve a request when classifier confidence alone would not.

For supervised tasks, deterministic rules extract a target phrase or an explicitly named target column. The column matcher compares the phrase with dataset columns using lexical matching and dataset context. Optional local embedding and reranking models add semantic matching when available. A match is accepted only when it meets the configured score and separation thresholds; otherwise the candidate columns are returned for clarification. Clustering does not require a target.

The request router can distinguish tabular and image requests when the request and asset context are consistent. Ambiguous or conflicting modality requests are not silently treated as a confirmed task.

## Dataset intelligence and preprocessing

Tabular upload returns validation and analysis information, including dimensions, data types, missing values, and sample data. Dataset intelligence identifies likely identifier, constant, and high-cardinality columns and provides recommendations. Training preparation excludes configured target/group columns and detected identifier or constant features, imputes missing values, and encodes categorical features; fitted preprocessing information is retained with the trained model for prediction.

Clustering has a separate mixed-type preprocessing pipeline. It can represent numeric and reliably parsed date features, encode categorical values, and report features it excludes, such as likely identifiers, constant columns, or unparseable date-like fields.

## AutoML algorithms and evaluation

The supervised tabular baseline candidate families are:

| Task | Candidate algorithms |
| --- | --- |
| Classification | Logistic Regression, Decision Tree Classifier, Random Forest Classifier |
| Regression | Linear Regression, Decision Tree Regressor, Random Forest Regressor |

Each candidate is trained and evaluated; per-model failures are recorded so other candidates can still be considered. Training supports grid search by default, and request options include automatic or Bayesian optimization. Automatic optimization is limited to supervised classification and regression and selects a bounded strategy based on available data, cross-validation capacity, model support, and optional Optuna availability. Bayesian optimization requires the optional Optuna package.

Reported supervised classification metrics include accuracy, weighted and macro precision/recall/F1, balanced accuracy, confusion matrix, and per-class metrics. Model selection uses weighted F1 for balanced training labels and balanced accuracy when the largest-to-smallest class ratio reaches the configured imbalance threshold.

Regression reports MAE, RMSE, and R²; selection uses the lowest RMSE. Evaluation is separated from the training/validation-based model-selection process.

### Clustering

Clustering evaluates K-Means and BIRCH candidates with cluster counts from 2 through the smaller of the square root of the training sample size and the configured 12-candidate cap. Candidate scoring uses a deterministic sample capped at 3,000 rows for large datasets. The system reports silhouette, Calinski–Harabasz, and Davies–Bouldin scores and cluster sizes; selection uses the mean rank across the three clustering metrics. Because these are geometric cluster-quality measures, they do not establish the real-world usefulness of a segmentation.

The supported saved clustering models are prediction-compatible: they assign new rows through the persisted fitted preprocessing and clusterer. The code does not offer DBSCAN or agglomerative clustering in this saved-model prediction workflow.

### Image classification

The image service trains a MobileNetV2 classifier from ZIP datasets organized by class. Training can use pretrained weights when available and records whether weights were loaded. Image analysis, training, and prediction are separate from the tabular AutoML algorithm set.

## Model persistence and predictions

Tabular training saves the selected model and its Model ID with the target, problem type, feature schema, encoders, imputation/preprocessing metadata, selection metric, metrics, and training configuration. By default, tabular packages are stored under `storage/models/`; the Compose service mounts `storage/` for persistence. The loader also checks the legacy `backend/storage/models/` directory when using its default storage location.

Use a tabular Model ID with `POST /predict/` and a list of feature records, or with `POST /predict/file` and a CSV/XLSX upload. Input must include the model's required raw feature columns; extra file columns are ignored. The saved preprocessing and model are applied without refitting during inference. The prediction response includes predictions and, where supported, class probabilities. The frontend can export prediction results as CSV.

Image models use a separate package store under `storage/image_models/`. Use the returned image Model ID with `POST /image/predict` and a single supported image file.

## Training-time estimates and reports

Before tabular training, the NLP response may include a deterministic heuristic training-time range derived from row count, feature count, problem type, and candidate count. It is an approximation, not a runtime guarantee; actual duration depends on the data, hardware, and tuning path. Completed training also records measured duration.

The NLP training workflow returns a JSON-safe training report with task and target, dataset and feature counts, trained/failed model counts, selected model, selection metric, evaluation metrics, Model ID, and warnings. Clustering reports also include clustering evaluation details. A result-explanation component uses `google/flan-t5-small` when available to explain supplied training facts and falls back to deterministic factual text if the model cannot be loaded. **A BART-based explanation layer remains planned and is not implemented.**

## Design principles

- **Clarify rather than guess:** unknown or ambiguous intents and target matches should not silently start the wrong supervised task.
- **Separate responsibilities:** API routing, dataset services, NLP resolution, AutoML, persistence, and prediction are separate components.
- **Reuse saved preprocessing:** prediction applies the preprocessing fitted during training rather than refitting on new data.
- **Keep results inspectable:** model metrics, failed candidates, feature exclusions, reports, and readiness checks are exposed to help users understand outcomes.

## Testing

Python backend tests use pytest. From the repository root:

```powershell
python -m pytest -q
```

The frontend includes Node's built-in test-runner tests and defines its test command in `frontend/package.json`:

```powershell
cd frontend
npm test
```

The tests cover areas including API workflows, NLP resolution and routing, training and prediction, persistence, clustering, reports, and image classification. No single test result or model-accuracy figure is asserted here; run the tests in the environment you intend to use.

## Limitations and future scope

### Current limitations

- The required intent classifier is a local artifact; `/ready` remains unavailable until a complete model and label mapping are configured.
- Semantic target-matching model files are optional for readiness, but requests that require them cannot use semantic matching until they are available in the local Hugging Face cache.
- Training-time estimates are heuristic, and model quality depends on dataset quality, task formulation, data split, and runtime environment.
- Clustering metrics describe geometric separation, not domain-specific segment value.
- Image classification is a dedicated classifier workflow, not general-purpose image understanding.
- The BART explanation layer is not implemented; the current result explanation is separate and uses a FLAN-T5 model with a factual fallback.

### Future scope

Potential extensions, not claims of current support, include broader prediction-compatible clustering methods, improved and more diverse intent training data, more robust semantic target matching, expanded model families and optimization, richer visual reporting, model monitoring, and the planned BART-based explanation layer.

## Additional documentation

- [Local deployment](docs/deployment.md)
- [Model setup](docs/model-setup.md)

## Author

Manasi Jadhav — AI & Data Science final-year project
