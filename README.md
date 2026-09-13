TextToAutoML

Natural Language Driven Machine Learning Automation

TextToAutoML is a natural-language-driven machine learning automation
system designed to make machine learning easier for non-technical users.
Users can upload a CSV/XLSX dataset and describe their ML task in plain
English. The system interprets the request, resolves the ML task and
target, runs the supported AutoML pipeline, evaluates candidate models,
saves the best model, and exposes it for prediction.

Example: "Predict whether customers will leave."

Project Status

The current implementation provides an end-to-end flow for
classification and regression. Clustering is recognized by the NLP
and recommendation layers, but clustering model training is not
currently implemented in the AutoML training pipeline.

Dataset + Natural Language Request
                ↓
          NLP Intent Detection
                ↓
       Confidence / Clarification
                ↓
            Task Mapping
                ↓
          Target Extraction
                ↓
          Target Matching
                ↓
        Dataset Resolution
                ↓
          AutoML Training
                ↓
        Model Evaluation
                ↓
         Best Model Selection
                ↓
          Model Persistence
                ↓
             Prediction

Key Features

CSV and XLSX/Excel dataset upload

Dataset validation, analysis, and preprocessing

Missing-value and duplicate handling

Date-column processing and feature extraction

Dataset intelligence for identifier, constant, and high-cardinality
columns

Natural-language ML task understanding

Fine-tuned DistilBERT intent classifier

Four intents: classification, regression, clustering, unknown

Confidence-based clarification for uncertain requests

Natural-language target extraction and deterministic target matching

Dataset-aware target resolution

Automated classification and regression model training

Model evaluation and best-model selection

.joblib model persistence

Prediction API with saved preprocessing and feature validation

Training-time estimation

Structured training reports

React + Vite frontend

FastAPI backend

NLP Intent Recognition

The NLP module uses:

distilbert-base-uncased

The intent dataset contains 200 balanced examples:

Intent             Examples

Classification           50
Regression               50
Clustering               50
Unknown                  50

The data is split into:

70% Training
15% Validation
15% Testing

Official held-out performance

Metric               Score

Accuracy            83.33%
Macro Precision     85.07%
Macro Recall        83.93%
Macro F1            82.98%

These are the currently measured results. The original 85% project
accuracy goal has not been claimed as achieved.

Clarification System

The clarification layer checks the top intent probability, the
difference between the top two probabilities, and whether the predicted
intent is unknown.

Current baseline thresholds:

Minimum top probability = 0.30
Minimum top-two margin  = 0.05

If a request is unknown, ambiguous, or insufficiently confident, the
system asks for clarification instead of training.

For example:

"Analyze this dataset"

does not specify a clear ML task, so the system requests clarification.

Target Resolution

For supervised learning, target understanding is separated into three
stages.

Target extraction

The system extracts the natural-language target:

Predict customer churn
        ↓
customer churn

Target matching

The extracted target is compared with actual dataset columns using
deterministic techniques:

normalization

exact matching

token overlap

containment

candidate scoring

ambiguity detection

The matcher deliberately avoids embeddings, fuzzy matching,
domain-specific synonym tables, and dataset-value inspection.

Dataset resolution

The final resolution combines the NLP intent, target information, and
dataset information. If the system cannot safely resolve the request, it
asks for clarification rather than making an unsafe assumption.

AutoML Pipeline

Dataset
   ↓
Target Column
   ↓
Problem Type
   ↓
Preprocessing
   ↓
Candidate Models
   ↓
Training
   ↓
Evaluation
   ↓
Best Model Selection
   ↓
Persistence

Classification models

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Regression models

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Clustering status

Clustering is currently supported at the intent/recommendation level.
The recommendation layer can identify clustering and recommend K-Means,
DBSCAN, or Agglomerative Clustering, but these models are not currently
trained, evaluated, persisted, or exposed through the prediction
pipeline.

Prediction API

The prediction endpoint is:

POST /predict/

Example request:

{
  "model_id": "your_model_id",
  "data": [
    {
      "feature_1": 10,
      "feature_2": "value"
    }
  ]
}

The prediction pipeline loads the saved model, validates required
features, applies the saved preprocessing and feature order, and returns
JSON-safe predictions. Classification models can also return
probabilities when supported.

Training-Time Estimation

TextToAutoML includes a lightweight training-time estimator based on:

number of rows

number of input features

problem type

number of candidate models

It provides an estimated time plus a minimum and maximum range. This is
a rough user-facing estimate, not a guaranteed execution time; actual
time depends on hardware, dataset characteristics, model complexity, and
system load.

Training Reports

After successful NLP-driven training, the system generates a structured
report containing information such as:

task

target column

problem type

dataset size

input feature count

training/test rows

models trained and failed

selected model

selection metric

model metrics

model ID

warnings

A future planned layer is BART-based natural-language reporting,
which will turn structured AutoML results into a simple explanation and
model usage guide.

Backend API

The backend uses FastAPI.

Endpoint                            Purpose

POST /upload/                     Upload and analyze a dataset

POST /train/                      Run the existing AutoML training
flow

POST /predict/                    Make predictions using a saved
model

POST /nlp/analyze                 Interpret a natural-language ML
request

Swagger documentation:

http://127.0.0.1:8001/docs

Frontend

The frontend uses React + Vite and currently supports:

Dataset upload

Dataset readiness information

Natural-language task input

NLP analysis

Clarification messages

Target/problem-type display

Training confirmation

Training status

Training results

Development URL:

http://127.0.0.1:5173/

Project Structure

TextToAutoML/
│
├── backend/
│   ├── api/
│   │   ├── upload.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── report.py
│   │   └── nlp.py
│   │
│   ├── services/
│   │   ├── dataset/
│   │   │   ├── loader.py
│   │   │   ├── validator.py
│   │   │   ├── analyze.py
│   │   │   ├── preprocess.py
│   │   │   └── intelligence.py
│   │   │
│   │   ├── automl/
│   │   │   ├── problem_detection.py
│   │   │   ├── pipeline.py
│   │   │   ├── trainer.py
│   │   │   ├── training.py
│   │   │   ├── evaluator.py
│   │   │   ├── persistence.py
│   │   │   ├── predictor.py
│   │   │   └── nlp_integration.py
│   │   │
│   │   ├── metalearning/
│   │   ├── reporting/
│   │   │   └── training_report.py
│   │   │
│   │   └── nlp/
│   │       ├── data/
│   │       │   ├── intent_dataset.csv
│   │       │   └── README.md
│   │       ├── intent_detection.py
│   │       ├── preprocessing.py
│   │       ├── task_mapping.py
│   │       ├── train_intent_model.py
│   │       ├── clarification.py
│   │       ├── target_extraction.py
│   │       ├── target_matching.py
│   │       ├── dataset_resolution.py
│   │       └── pipeline.py
│   │
│   └── storage/
│       ├── uploads/
│       ├── models/
│       ├── reports/
│       └── nlp_models/
│
├── frontend/
│   ├── package.json
│   ├── index.html
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       └── App.jsx
│
└── tests/

Technology Stack

Backend: Python, FastAPI, Pandas, NumPy, Scikit-learn, Joblib

NLP: Hugging Face Transformers, DistilBERT, PyTorch, Scikit-learn
metrics

Frontend: React, Vite, JavaScript

Data formats: CSV, XLSX

Model persistence: Joblib

Installation and Running

1. Clone the repository

git clone <repository-url>
cd TextToAutoML

2. Activate the Python environment

Windows:

.venv\Scripts\activate

Install the project's Python dependencies according to its dependency
configuration.

3. Start the backend

From the project root:

.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --reload --port 8001

Backend:

http://127.0.0.1:8001

4. Start the frontend

In another terminal:

cd frontend
npm install
npm run dev

Frontend:

http://127.0.0.1:5173/

Example Workflows

Classification

Upload Customer-Churn-Records.csv and enter:

Predict whether customers will leave

The system can resolve:

Intent       → Classification
Target       → Exited
Problem Type → Binary Classification

Regression

Upload housing.csv and enter:

Predict median house value

The system resolves:

Intent       → Regression
Target       → median_house_value
Problem Type → Regression

Unknown request

Enter:

Analyze this dataset

Expected behavior:

Intent              → Unknown
Needs clarification → Yes
Ready for training  → No
Training estimate   → None

Testing

Run the Python tests from the project root:

.venv\Scripts\python.exe -m pytest -q

The test suite includes coverage for NLP intent processing,
clarification, target extraction, target matching, dataset resolution,
NLP-to-AutoML integration, training reports, and training-time
estimation.

Design Principles

Separation of responsibilities

NLP determines what the user means; AutoML handles model training.

NLP
 ↓
Structured ML Task
 ↓
AutoML

Do not blindly select targets

Automatic dataset-based target detection can produce incorrect
candidates. Target resolution therefore combines the natural-language
target, dataset column matching, confidence, and clarification.

Fail safely

When intent or target resolution is uncertain, the system clarifies
instead of guessing.

Preserve the working AutoML core

The existing training and prediction pipeline remains a separate
component while NLP is integrated around it.

Current Limitations

DistilBERT currently achieves 83.33% official held-out accuracy, so
the intent model can still be improved.

The intent dataset contains only 200 examples.

Intent confidence thresholds are baseline heuristics rather than
calibrated probabilities.

Natural-language target matching is intentionally conservative.

Some natural-language descriptions may require clarification.

Clustering is recognized but does not yet have an implemented AutoML
training pipeline.

The training-time estimator is heuristic.

BART-based natural-language report generation is planned, but is not
yet part of the implemented training flow.

Future Scope

Expand the intent dataset with more real-world queries.

Improve intent classification and probability calibration.

Improve general natural-language target matching.

Implement clustering training, evaluation, persistence, and
prediction.

Strengthen model recommendation/meta-learning.

Improve hyperparameter optimization.

Add BART-based natural-language training reports and model usage
guidance.

Improve visualizations and user experience.

Expand testing across more datasets and real user requests.

Project Goal

The long-term goal of TextToAutoML is to make machine learning
accessible through natural language:

Instead of:
"Which algorithm should I use?"

The user says:
"Predict whether these customers will leave."

TextToAutoML handles the ML workflow.

The project aims to bridge the gap between natural-language user
requirements and automated machine learning pipelines, reducing the
technical knowledge required to build and use machine learning models.

Author

Manasi Umesh Jadhav
B.Tech Artificial Intelligence and Data Science
K. K. Wagh Institute of Engineering Education and Research

License

Add the project's chosen license here before public distribution.