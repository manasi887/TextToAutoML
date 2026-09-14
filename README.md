# TextToAutoML

### Natural Language Driven Machine Learning Automation

> **TextToAutoML** is an end-to-end machine learning automation system that allows users to describe their machine learning task in natural language and automatically connect that request to an AutoML training pipeline.

Instead of requiring users to manually determine the machine learning problem, select a target column, preprocess the dataset, choose algorithms, train models, and evaluate results, TextToAutoML aims to automate this workflow through a combination of **Natural Language Processing (NLP)** and **Automated Machine Learning (AutoML)**.

---

##  Project Status

| Component | Status |
|---|---|
| Dataset Upload |  Implemented |
| Dataset Validation |  Implemented |
| Dataset Analysis |  Implemented |
| Dataset Intelligence |  Implemented |
| NLP Intent Detection |  Implemented |
| Intent Clarification |  Implemented |
| Target Extraction |  Implemented |
| Target Matching |  Implemented |
| Target Resolution |  Implemented |
| Classification Training |  Implemented |
| Regression Training |  Implemented |
| Model Selection |  Implemented |
| Model Persistence |  Implemented |
| Prediction API |  Implemented |
| Training-Time Estimation |  Implemented |
| Training Report |  Implemented |
| React Frontend |  Implemented |
| NLP → AutoML Integration |  Implemented |
| Clustering Training | ⏳ Planned |
| BART User-Facing Explanation | ⏳ Planned |

---

#  Overview

Traditional AutoML systems generally require the user to understand concepts such as:

- Classification vs. regression
- Target-column selection
- Data preprocessing
- Algorithm selection
- Model evaluation
- Prediction input requirements

TextToAutoML introduces a **Natural Language Interface** on top of the AutoML workflow.

The user can provide a request such as:

> **"Predict whether customers will leave."**

TextToAutoML analyzes the request, identifies the intended machine learning task, resolves the target column from the uploaded dataset, estimates the training time, and can then pass the confirmed task to the existing AutoML pipeline.

---

#  Project Objective

The main objective of TextToAutoML is to create a system where users can interact with machine learning through **natural language rather than manually configuring an ML pipeline**.

The system focuses primarily on:

1. Natural-language understanding
2. Intent recognition
3. Target identification
4. Dataset analysis
5. Automated model training
6. Model evaluation
7. Model persistence
8. Prediction
9. Human-readable reporting

---

#  Core Idea

The overall concept can be represented as:

```text
                         USER
                           │
                           ▼
               Natural Language Request
                           │
                           ▼
                 ┌─────────────────┐
                 │    NLP Layer    │
                 │    DistilBERT   │
                 └────────┬────────┘
                          │
                          ▼
                  Intent Recognition
                          │
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
             Confident         Ambiguous /
              Intent             Unknown
                 │                 │
                 │                 ▼
                 │            Clarification
                 │                 │
                 └────────┬────────┘
                          │
                          ▼
                 Task / Target Resolution
                          │
                          ▼
                 Dataset Intelligence
                          │
                          ▼
                     AutoML Layer
                          │
                          ▼
               Model Training & Evaluation
                          │
                          ▼
                     Best Model
                          │
                  ┌───────┴────────┐
                  ▼                ▼
            Training Report     Prediction
                  │
                  ▼
           BART Explanation
              (Planned)
````

---

#  End-to-End Workflow

```text
1. Upload Dataset
       ↓
2. Validate Dataset
       ↓
3. Analyze Dataset
       ↓
4. Enter Natural Language Request
       ↓
5. Detect User Intent
       ↓
6. Check Intent Confidence
       ↓
7. Ask for Clarification if Required
       ↓
8. Map Intent to ML Task
       ↓
9. Extract Target Reference
       ↓
10. Match Target to Dataset Column
       ↓
11. Resolve Final Task + Target
       ↓
12. Estimate Training Time
       ↓
13. Run AutoML
       ↓
14. Train Candidate Models
       ↓
15. Evaluate Models
       ↓
16. Select Best Model
       ↓
17. Save Model
       ↓
18. Generate Training Report
       ↓
19. Explain Model to User
       ↓
20. Predict on New Data
```

---

#  System Architecture

TextToAutoML is divided into several major layers.

```text
┌─────────────────────────────────────────────────────┐
│                    React Frontend                   │
│                                                     │
│ Dataset Upload → Task Input → Results               │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                    FastAPI API                      │
│                                                     │
│ /upload/                                            │
│ /nlp/analyze                                        │
│ /nlp/train                                          │
│ /train/                                              │
│ /predict/                                            │
└────────────────┬────────────────────┬───────────────┘
                 │                    │
                 ▼                    ▼
┌─────────────────────────┐  ┌────────────────────────┐
│       NLP Layer         │  │     Dataset Layer      │
│                         │  │                        │
│ DistilBERT              │  │ Loader                 │
│ Intent Detection        │  │ Validator              │
│ Clarification           │  │ Analyzer               │
│ Target Extraction       │  │ Preprocessing          │
│ Target Matching         │  │ Intelligence           │
└────────────┬────────────┘  └────────────┬───────────┘
             │                            │
             └──────────────┬─────────────┘
                            ▼
                 ┌─────────────────────┐
                 │     AutoML Layer    │
                 │                     │
                 │ Problem Detection   │
                 │ Pipeline            │
                 │ Training            │
                 │ Evaluation          │
                 │ Model Selection     │
                 │ Persistence         │
                 │ Prediction          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Reporting Layer   │
                 │                     │
                 │ Training Report     │
                 │ BART Explanation    │
                 │      (Planned)      │
                 └─────────────────────┘
```

---

#  Project Structure

```text
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
│   │   │   ├── nlp_integration.py
│   │   │   └── time_estimator.py
│   │   │
│   │   ├── metalearning/
│   │   │
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
│   ├── storage/
│   │   ├── uploads/
│   │   ├── models/
│   │   ├── reports/
│   │   └── nlp_models/
│   │       └── intent_classifier/
│   │
│   └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   └── App.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── tests/
│   ├── test_nlp_pipeline.py
│   └── test_time_estimator.py
│
├── README.md
└── ...
```

---

#  Natural Language Processing Layer

The NLP layer is the main interface between the user's natural-language request and the machine learning system.

It performs:

- Intent detection
- Confidence analysis
- Clarification
- Task mapping
- Target extraction
- Target-column matching
- Dataset resolution

---

## Intent Recognition

TextToAutoML uses **DistilBERT** for intent classification.

The model recognizes four intent classes:

| IntentMeaning    |                                                            |
| ---------------- | ---------------------------------------------------------- |
| `classification` | Predict a category or class                                |
| `regression`     | Predict a numerical value                                  |
| `clustering`     | Group similar records                                      |
| `unknown`        | Request cannot be confidently mapped to a supported intent |

The model is based on:

```text
distilbert-base-uncased
```

---

#  Intent Dataset

The current intent dataset contains:

```text
Total examples: 200

Classification: 50
Regression:     50
Clustering:     50
Unknown:        50
```

The dataset is balanced across all four intent classes.

The examples are designed to represent realistic natural-language machine learning requests.

---

#  NLP Model Evaluation

The current held-out test performance of the DistilBERT intent classifier is:

| MetricScore     |        |
| --------------- | ------ |
| Accuracy        | 83.33% |
| Macro Precision | 85.07% |
| Macro Recall    | 83.93% |
| Macro F1        | 82.98% |

### Per-Intent Performance

| IntentPrecisionRecallF1 |      |      |      |
| ----------------------- | ---- | ---- | ---- |
| Classification          | 1.00 | 0.62 | 0.77 |
| Regression              | 0.75 | 0.86 | 0.80 |
| Clustering              | 0.88 | 0.88 | 0.88 |
| Unknown                 | 0.78 | 1.00 | 0.88 |

> **Note:** These are the official held-out evaluation results. A separate handpicked diagnostic set produced a higher result, but it is not used as the official model evaluation.

---

#  Intent Confidence & Clarification

TextToAutoML does not blindly trust the intent classifier.

The system checks:

- Top intent probability
- Difference between the top two intents
- Whether the predicted intent is `unknown`

The current baseline clarification rules are:

```text
Minimum top probability = 0.30
Minimum top-two margin  = 0.05
```

If the request is uncertain, the system asks the user to clarify instead of continuing automatically.

### Example

Input:

```text
Analyze this dataset
```

Output:

```text
Intent: unknown
Needs clarification: true
Ready for training: false
Training time estimate: null
```

This prevents an ambiguous request from accidentally triggering an ML task.

---

#  Target Extraction

After identifying a supported intent, TextToAutoML attempts to determine what the user wants to predict.

For example:

```text
Predict customer churn
```

becomes:

```text
Target reference:
customer churn
```

Another example:

```text
Predict median house value
```

becomes:

```text
Target reference:
median house value
```

Target extraction is currently implemented using deterministic rules rather than another ML model.

---

#  Target Matching

The extracted natural-language target is matched against the dataset's actual column names.

The matcher uses:

- Normalization
- Exact matching
- Token overlap
- Containment
- Candidate scoring
- Confidence thresholds
- Ambiguity detection

Current matching thresholds include:

```text
Minimum match score       = 0.80
Minimum score margin      = 0.25
Minimum candidate score   = 0.50
```

The system preserves the original dataset column name when returning the matched target.

---

#  Dataset Intelligence

The dataset layer provides information required for automated decision-making.

## Dataset Loading

Supports:

- CSV
- XLSX

## Dataset Validation

Checks for problems such as:

- Empty datasets
- Invalid input
- Missing target columns
- Dataset structure issues

## Dataset Analysis

Provides information such as:

- Number of rows
- Number of columns
- Data types
- Missing values
- Dataset characteristics

## Dataset Intelligence

Detects:

- Identifier columns
- Constant columns
- High-cardinality columns
- Potentially useful dataset recommendations

---

#  AutoML Layer

The AutoML layer receives the resolved machine learning task and dataset.

The current supervised training workflow supports:

- Binary Classification
- Multi-class Classification
- Regression

---

## Supported Models

### Classification

```text
LogisticRegression
DecisionTreeClassifier
RandomForestClassifier
```

### Regression

```text
LinearRegression
DecisionTreeRegressor
RandomForestRegressor
```

The models are trained and evaluated automatically.

The best-performing model is selected according to the task-specific evaluation metric.

---

#  Model Evaluation

## Classification

The classification pipeline evaluates models using metrics including:

- Accuracy
- Precision
- Recall
- F1 Score

Model selection uses:

```text
F1 Score
```

## Regression

The regression pipeline evaluates models using metrics including:

- RMSE
- R²

Model selection uses:

```text
RMSE
```

Lower RMSE is better.

---

#  Model Persistence

After training, the selected model is persisted as a model package.

The package stores information such as:

```text
Model
Model ID
Target column
Problem type
Model name
Metrics
Preprocessing information
Feature names
Creation timestamp
```

This allows the saved model to be reused later for predictions.

---

#  Prediction

The prediction API accepts:

- Model ID
- New input records

The system loads the saved model and applies the same preprocessing configuration used during training.

The prediction response can contain:

```text
Prediction
Prediction count
Class probabilities
Model name
Problem type
Target column
```

For classification models, prediction probabilities are also returned when supported.

---

# Training-Time Estimation

TextToAutoML provides an estimated training time before training begins.

The estimator considers:

- Number of rows
- Number of features
- Problem type
- Number of candidate models

The current estimator is a lightweight heuristic intended to provide users with an approximate expectation rather than an exact runtime guarantee.

Conceptually:

```text
Dataset Size
     +
Feature Count
     +
Model Count
     +
Problem Type
     ↓
Estimated Training Time
```

The frontend displays the result in a human-readable format such as:

```text
Estimated training time:
6 seconds – 9 seconds
```

The actual training process also records the measured training duration.

---

#  Training Report

After successful training, TextToAutoML generates a structured training report.

The report contains information such as:

## Task

- Target column
- Problem type

## Dataset

- Number of rows
- Number of usable input features
- Training rows
- Testing rows

## Models

- Number of models trained
- Number of failed models

## Best Model

- Model name
- Selection metric
- Evaluation metrics
- Model ID

The report is designed to provide a concise summary of the completed AutoML process.

---

#  BART User-Facing Explanation

A planned component of TextToAutoML is a **BART-based explanation layer**.

BART is intended to work **after AutoML**, rather than making the machine learning decisions itself.

The planned architecture is:

```text
User Request
     ↓
DistilBERT
     ↓
Task + Target Resolution
     ↓
AutoML
     ↓
Best Model + Metrics + Required Features
     ↓
Structured Training Report
     ↓
BART
     ↓
Human-Readable Explanation
     ↓
How to Use the Model
```

The purpose of BART will be to transform structured AutoML results into an understandable explanation for non-technical users.

For example, it may explain:

- What task was performed
- Which target was predicted
- Which model was selected
- How well the model performed
- Which input fields are required
- How the user can provide new data for prediction

> BART is currently a **planned feature** and is not yet part of the implemented production pipeline.

---

#  Clustering Status

Clustering is recognized by the NLP layer.

For example:

```text
Group customers into similar segments
```

can be identified as:

```text
Intent:
clustering
```

However, the current AutoML training layer does **not yet implement clustering model training**.

Currently supported AutoML training types are:

```text
Binary Classification
Multi-class Classification
Regression
```

Therefore, clustering is currently supported at the **NLP/task-recognition level**, while clustering training remains future scope.

Potential future algorithms include:

```text
K-Means
DBSCAN
Agglomerative Clustering
```

---

#  FastAPI Backend

The backend is implemented using **FastAPI**.

## Main API Endpoints

| EndpointMethodPurpose |          |                                      |
| --------------------- | -------- | ------------------------------------ |
| `/upload/`            | POST     | Upload and analyze a dataset         |
| `/nlp/analyze`        | POST     | Analyze a natural-language request   |
| `/nlp/train`          | POST     | Resolve NLP request and train AutoML |
| `/train/`             | POST     | Direct AutoML training               |
| `/predict/`           | POST     | Generate predictions                 |
| `/report/`            | GET/POST | Training report functionality        |

FastAPI also provides interactive API documentation through Swagger UI.

---

#  Frontend

The frontend is implemented using:

- React
- Vite
- JavaScript

The current frontend provides a simple workflow:

```text
Upload Dataset
      ↓
Enter Natural Language Task
      ↓
Analyze
      ↓
View Intent / Target / Problem Type
      ↓
Confirm and Train
      ↓
View Training Result
```

The interface also handles clarification requests.

For example, if the system receives:

```text
Analyze this dataset
```

the frontend displays a clarification message instead of starting training.

---

#  Frontend ↔ Backend

During development, Vite proxies frontend API requests to the FastAPI backend.

```text
React / Vite
     │
     │ HTTP
     ▼
FastAPI
     │
     ├── /upload/
     ├── /nlp/analyze
     ├── /nlp/train
     ├── /train/
     └── /predict/
```

---

#  Technology Stack

| AreaTechnology          |                           |
| ----------------------- | ------------------------- |
| Frontend                | React                     |
| Frontend Build Tool     | Vite                      |
| Backend                 | FastAPI                   |
| Programming Language    | Python                    |
| NLP                     | Hugging Face Transformers |
| NLP Model               | DistilBERT                |
| NLP Utilities           | SpaCy                     |
| Machine Learning        | Scikit-learn              |
| Data Processing         | Pandas                    |
| Deep Learning Framework | PyTorch                   |
| Model Serialization     | Joblib                    |
| API Documentation       | Swagger / OpenAPI         |
| Version Control         | Git / GitHub              |
| Containerization        | Docker                    |
| Database                | PostgreSQL                |

---

#  Installation

## 1. Clone the Repository

```bash
git clone https://github.com/manasi887/TextToAutoML.git
cd TextToAutoML
```

---

## 2. Create a Python Virtual Environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install Backend Dependencies

Install the required Python packages according to the project's dependency configuration.

If a `requirements.txt` file is available:

```powershell
pip install -r requirements.txt
```

---

# Running the Backend

From the project root:

```powershell
.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --reload --port 8001
```

The backend will be available at:

```text
http://127.0.0.1:8001
```

Swagger documentation:

```text
http://127.0.0.1:8001/docs
```

---

# Running the Frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Vite will provide the local frontend URL.

---

#  Testing

The project contains focused automated tests for important components.

For example:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_nlp_pipeline.py tests/test_time_estimator.py backend/test_nlp_integration.py
```

Current verified result:

```text
13 passed in 2.68s
```

---

#  Example Workflow 1 — Classification

## Dataset

```text
Customer-Churn-Records.csv
```

## User Request

```text
Predict whether customers will leave
```

## NLP Resolution

```text
Intent:
classification

Target:
Exited

Problem Type:
Binary Classification

Ready for Training:
true
```

## AutoML

The system can then train candidate classification models and select the best-performing model.

---

#  Example Workflow 2 — Regression

## Dataset

```text
housing.csv
```

## User Request

```text
Predict median house value
```

## NLP Resolution

```text
Intent:
regression

Target:
median_house_value

Problem Type:
Regression

Ready for Training:
true
```

The AutoML layer then evaluates the available regression models and selects the best-performing model.

---

#  Example Workflow 3 — Unknown Request

## User Request

```text
Analyze this dataset
```

## Result

```text
Intent:
unknown

Needs Clarification:
true

Ready for Training:
false

Training Time Estimate:
null
```

The system does **not** automatically select a machine learning task.

Instead, it asks the user to clarify what they want to accomplish.

This is an important safety mechanism in the NLP-to-AutoML workflow.

---

#  Design Principles

TextToAutoML follows several important design principles.

## 1. Natural Language First

Users should be able to describe their ML goal naturally.

## 2. Don't Guess When Uncertain

Ambiguous requests should trigger clarification instead of silently selecting an incorrect task.

## 3. Deterministic Target Resolution

Target matching uses explicit scoring and ambiguity rules rather than uncontrolled assumptions.

## 4. Separation of Responsibilities

The system separates:

```text
NLP
Dataset Intelligence
AutoML
Reporting
Prediction
```

This makes the architecture easier to test and maintain.

## 5. Reuse Existing AutoML

The NLP layer does not duplicate the existing AutoML training logic.

Instead:

```text
NLP
 ↓
Confirmed Resolution
 ↓
Existing AutoML Pipeline
```

## 6. JSON-Safe API Responses

API responses are normalized so that model outputs and numerical values can safely be serialized into JSON.

---

#  Current Limitations

The current implementation has several known limitations.

## NLP Model Performance

The current held-out DistilBERT accuracy is:

```text
83.33%
```

Therefore, the NLP model should not currently be described as achieving 90%+ official test accuracy.

## Training-Time Estimation

The current training-time estimator is heuristic.

Actual training time depends on:

- Hardware
- CPU/GPU availability
- Dataset characteristics
- Model complexity
- System load

Therefore, the estimate should be treated as approximate.

## Clustering

Clustering is recognized by the NLP layer but is not yet implemented in the AutoML training layer.

## BART

The BART explanation layer is planned and has not yet been integrated into the production workflow.

## Target Matching

Target matching currently relies on deterministic matching techniques and may require clarification for highly ambiguous or semantically different target descriptions.

---

#  Future Scope

Future development can extend TextToAutoML with:

- Full clustering support
- K-Means / DBSCAN / Agglomerative Clustering
- Improved NLP intent classification
- Larger and more diverse intent datasets
- Better confidence calibration
- Advanced semantic target matching
- Meta-learning based algorithm recommendation
- Hyperparameter optimization
- BART-based user explanations
- Richer visualizations
- Automated model comparison dashboards
- More sophisticated training-time prediction
- Expanded prediction workflows
- Improved model monitoring
- Dockerized deployment
- PostgreSQL-backed experiment tracking

---

#  Long-Term Vision

The long-term goal is to make machine learning accessible to users who may not have a strong machine learning background.

The intended experience is:

```text
"I have a dataset and I want to know
which customers are likely to leave."

                    ↓

              TextToAutoML

                    ↓

        Understand the request

                    ↓

        Understand the dataset

                    ↓

        Determine the ML problem

                    ↓

        Identify the target

                    ↓

        Train multiple models

                    ↓

        Select the best model

                    ↓

        Explain the result

                    ↓

        Make predictions
```

The system aims to transform a traditionally technical ML workflow into a more natural, guided interaction.

---

#  Author

**Manasi Jadhav**

B.Tech — Artificial Intelligence & Data Science

**TextToAutoML — Final Year Project**

---

#  License

This project is currently developed as an academic Final Year Project.

A formal open-source license can be added when the project is prepared for public distribution.

---

# Project Summary

**TextToAutoML** combines:

```text
Natural Language Processing
            +
Dataset Intelligence
            +
Automated Machine Learning
            +
Model Evaluation
            +
Model Persistence
            +
Prediction
            +
Human-Readable Reporting
```

to create a natural-language-driven machine learning automation workflow.

> **Describe your machine learning goal.**
> **Let TextToAutoML handle the pipeline.**

```