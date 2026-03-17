# MLOps House Price Prediction (Beginner Friendly)

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Final Project Flow (Architecture Diagram)](#2-final-project-flow-architecture-diagram)
3. [Tools Used (What + Why)](#3-tools-used-what--why)
4. [Folder Structure](#4-folder-structure)
5. [Step 1 — Create Project Folder](#5-step-1--create-project-folder)
6. [Step 2 — Create Python Virtual Environment](#6-step-2--create-python-virtual-environment)
7. [Step 3 — Install Required Packages](#7-step-3--install-required-packages)
8. [Step 4 — Initialize Git](#8-step-4--initialize-git)
9. [Step 5 — Create Simple Dataset](#9-step-5--create-simple-dataset)
10. [Step 6 — Create Preprocessing Script](#10-step-6--create-preprocessing-script)
11. [Step 7 — Create Hyperparameter Config File](#11-step-7--create-hyperparameter-config-file)
12. [Step 8 — Create Training Script](#12-step-8--create-training-script)
13. [Step 9 — Run Preprocessing and Training](#13-step-9--run-preprocessing-and-training)
14. [Step 10 — Start MLflow UI](#14-step-10--start-mlflow-ui)
15. [Step 11 — Create Prediction Helper Script](#15-step-11--create-prediction-helper-script)
16. [Step 12 — Create FastAPI App](#16-step-12--create-fastapi-app)
17. [Step 13 — Run the API](#17-step-13--run-the-api)
18. [Step 14 — Test Prediction API](#18-step-14--test-prediction-api)
19. [Step 15 — Initialize DVC](#19-step-15--initialize-dvc)
20. [Step 16 — Create DVC Pipeline](#20-step-16--create-dvc-pipeline)
21. [Step 17 — Run DVC Pipeline](#21-step-17--run-dvc-pipeline)
22. [Step 18 — Save Everything in Git](#22-step-18--save-everything-in-git)
23. [Mini Architecture Diagram (Cleanest Version)](#23-mini-architecture-diagram-cleanest-version)

---

# 1. Project Overview

## What are we building?

A **beginner-friendly MLOps project** that predicts **house prices** using:

* `size_sqft`
* `bedrooms`
* `location_score`

The project covers the **full ML lifecycle up to local API serving**:

* dataset creation
* preprocessing
* training
* evaluation
* model saving
* experiment tracking with **MLflow**
* data/model pipeline tracking with **DVC**
* serving the model with **FastAPI**
* versioning code with **Git**

---

## Why this project?

This project is simple enough for beginners, but still teaches **real MLOps concepts**.

You will understand:

* how data flows in ML
* how models are trained
* why experiment tracking matters
* why data versioning matters
* how to expose a model as an API
* how to structure a production-style ML project

---

# 2. Final Project Flow (Architecture Diagram)

```text
        ┌──────────────┐
        │   Dataset    │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ Preprocess   │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ Train Model  │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ Evaluate     │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ Save Model   │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ MLflow Track │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ FastAPI Serve│
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ DVC Pipeline │
        └──────────────┘
```

---

## Beginner meaning of the flow

1. **Dataset** → We need data to teach the model.
2. **Preprocess** → Clean and prepare the data.
3. **Train Model** → Let the algorithm learn patterns.
4. **Evaluate** → Check whether the model is good.
5. **Save Model** → Store the trained model for reuse.
6. **MLflow Track** → Save experiment parameters and metrics.
7. **FastAPI Serve** → Turn the model into an API.
8. **DVC Pipeline** → Make the pipeline reproducible and track changes.

---

# 3. Tools Used (What + Why)

| Tool         | What it is            | Why we use it                                |
| ------------ | --------------------- | -------------------------------------------- |
| Python       | Programming language  | Write ML and API code                        |
| Pandas       | Data library          | Read and process CSV files                   |
| Scikit-learn | ML library            | Train the house price model                  |
| MLflow       | Experiment tracker    | Track params, metrics, artifacts             |
| DVC          | Data Version Control  | Version data and build reproducible pipeline |
| FastAPI      | API framework         | Expose the model as REST API                 |
| Uvicorn      | ASGI server           | Run FastAPI locally                          |
| Joblib       | Serialization library | Save/load trained model                      |
| Git          | Version control       | Track code changes                           |

---

# 4. Folder Structure

```text
mlops-house-price/
│
├── data/
│   ├── raw/
│   │   └── house_data.csv
│   └── processed/
│       └── processed_house_data.csv
│
├── models/
│   └── model.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Why this structure?

* `data/raw/` → original data (never directly modify)
* `data/processed/` → cleaned data used by training
* `models/` → saved trained model
* `src/` → ML pipeline code
* `app/` → serving code (API)
* `params.yaml` → configuration (hyperparameters)
* `dvc.yaml` → reproducible pipeline stages
* `README.md` → project guide and documentation

---

# 5. Step 1 — Create Project Folder

## Command

```bash
mkdir mlops-house-price
cd mlops-house-price
```

## Why use it?

We start with a clean project directory so all files are organized in one place.

---

# 6. Step 2 — Create Python Virtual Environment

## Windows (PowerShell)

```bash
python -m venv venv
venv\Scripts\activate
```

## Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Why use it?

A virtual environment keeps this project’s dependencies isolated.

Without it:

* package conflicts happen
* different projects break each other

With it:

* reproducible setup
* clean local environment
* safer dependency management

---

# 7. Step 3 — Install Required Packages

## Create `requirements.txt`

```txt
pandas
scikit-learn
mlflow
fastapi
uvicorn
joblib
dvc
python-multipart
pyyaml
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Why each package?

* `pandas` → read and manipulate tabular data
* `scikit-learn` → train/test split and regression model
* `mlflow` → track ML experiments
* `fastapi` → create REST API
* `uvicorn` → run the API server
* `joblib` → save/load model files
* `dvc` → track data and pipeline stages
* `pyyaml` → read `params.yaml`

---

# 8. Step 4 — Initialize Git

## Command

```bash
git init
```

## Create `.gitignore`

```gitignore
venv/
__pycache__/
*.pyc
mlruns/
.dvc/cache/
models/
```

## Why use Git?

Git tracks **code changes**.

In MLOps:

* **Git** tracks code
* **DVC** tracks data and model artifacts

That separation is important.

---

# 9. Step 5 — Create Simple Dataset

## Create `data/raw/house_data.csv`

```csv
size_sqft,bedrooms,location_score,price
1000,2,7,50000
1200,2,8,60000
1500,3,8,75000
1800,3,9,90000
2000,4,9,110000
2200,4,10,130000
1400,3,7,70000
1600,3,8,80000
2500,5,10,150000
900,2,6,45000
```

## Why create this dataset?

This is our training data.

* **Features** = `size_sqft`, `bedrooms`, `location_score`
* **Label** = `price`

This dataset is intentionally simple so beginners can focus on MLOps, not complex ML.

---

# 10. Step 6 — Create Preprocessing Script

## File: `src/data_preprocessing.py`

```python
import pandas as pd
import os

RAW_DATA_PATH = "data/raw/house_data.csv"
PROCESSED_DATA_PATH = "data/processed/processed_house_data.csv"

def preprocess():
    df = pd.read_csv(RAW_DATA_PATH)

    # Simple preprocessing
    df = df.dropna()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess()
```

## What this script does

* reads raw CSV
* removes missing values
* writes cleaned CSV to `data/processed/`

## Why use preprocessing?

Real-world data is rarely clean.

Common issues:

* missing values
* duplicate rows
* wrong formats
* bad column values

Preprocessing makes the data safe for training.

---

# 11. Step 7 — Create Hyperparameter Config File

## File: `params.yaml`

```yaml
train:
  test_size: 0.2
  random_state: 42
```

## Why use `params.yaml`?

Instead of hardcoding settings inside Python, we keep them in a config file.

Benefits:

* easier to change
* easier to track
* better for experiments
* cleaner code

---

# 12. Step 8 — Create Training Script

## File: `src/train.py`

```python
import pandas as pd
import os
import joblib
import yaml
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

PROCESSED_DATA_PATH = "data/processed/processed_house_data.csv"
MODEL_PATH = "models/model.pkl"
PARAMS_PATH = "params.yaml"

def train():
    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f)

    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]

    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df[["size_sqft", "bedrooms", "location_score"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    mlflow.set_experiment("house-price-prediction")

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        mlflow.log_artifact(MODEL_PATH)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model saved to {MODEL_PATH}")
        print(f"MSE: {mse}")
        print(f"R2 Score: {r2}")

if __name__ == "__main__":
    train()
```

## What this script does

1. Reads config from `params.yaml`
2. Loads processed data
3. Splits features and label
4. Splits train/test
5. Trains `LinearRegression`
6. Evaluates with MSE and R²
7. Saves the model
8. Logs metrics and artifacts to MLflow

---

## Why use a training script?

This is the **core ML step**.

It transforms:

* input data
  into
* a trained reusable model

This is the main difference between raw data and an intelligent prediction system.

---

# 13. Step 9 — Run Preprocessing and Training

## Commands

```bash
python src/data_preprocessing.py
python src/train.py
```

## Why this order?

Training depends on processed data.

So:

1. preprocess first
2. then train

---

# 14. Step 10 — Start MLflow UI

## Command

```bash
mlflow ui
```

## Open in browser

```text
http://127.0.0.1:5000
```

## Why use MLflow UI?

It helps you visually inspect:

* experiments
* runs
* parameters
* metrics
* saved artifacts

Beginner understanding:

> MLflow is like a dashboard for your ML experiments.

---

# 15. Step 11 — Create Prediction Helper Script

## File: `src/predict.py`

```python
import joblib
import pandas as pd

MODEL_PATH = "models/model.pkl"

def predict(size_sqft, bedrooms, location_score):
    model = joblib.load(MODEL_PATH)

    input_data = pd.DataFrame([{
        "size_sqft": size_sqft,
        "bedrooms": bedrooms,
        "location_score": location_score
    }])

    prediction = model.predict(input_data)
    return prediction[0]

if __name__ == "__main__":
    result = predict(1700, 3, 8)
    print(f"Predicted Price: {result}")
```

## Why do we need this?

This is the simplest form of **inference**.

Training is done.
Now we use the saved model on new data.

---

# 16. Step 12 — Create FastAPI App

## File: `app/main.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="House Price Prediction API")

MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)

class HouseFeatures(BaseModel):
    size_sqft: float
    bedrooms: int
    location_score: float

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}

@app.post("/predict")
def predict(features: HouseFeatures):
    input_data = pd.DataFrame([{
        "size_sqft": features.size_sqft,
        "bedrooms": features.bedrooms,
        "location_score": features.location_score
    }])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }
```

## Why use FastAPI?

A saved `.pkl` model file is not directly useful to users.

FastAPI turns the model into a **REST API** so:

* browser clients can call it
* frontend apps can call it
* backend services can call it
* future Kubernetes deployments can expose it

---

# 17. Step 13 — Run the API

## Command

```bash
uvicorn app.main:app --reload
```

## Open Swagger docs

```text
http://127.0.0.1:8000/docs
```

## Why use Swagger docs?

FastAPI automatically creates an interactive UI.

This is perfect for beginners because you can test endpoints without Postman.

---

# 18. Step 14 — Test Prediction API

## Sample request body

```json
{
  "size_sqft": 1700,
  "bedrooms": 3,
  "location_score": 8
}
```

## Sample response

```json
{
  "predicted_price": 85000.0
}
```

## Why test locally first?

Before Docker, before Kubernetes, before CI/CD:

Always verify:

* model loads correctly
* API starts correctly
* endpoint returns prediction

This is a very important engineering habit.

---

# 19. Step 15 — Initialize DVC

## Command

```bash
dvc init
```

## Track dataset with DVC

```bash
dvc add data/raw/house_data.csv
```

## Why use DVC?

Git is excellent for code, but not ideal for:

* large datasets
* binary model artifacts
* reproducible ML pipelines

DVC solves that.

### Simple understanding

* **Git** = code version control
* **DVC** = data + model pipeline version control

---

# 20. Step 16 — Create DVC Pipeline

## File: `dvc.yaml`

```yaml
stages:
  preprocess:
    cmd: python src/data_preprocessing.py
    deps:
      - src/data_preprocessing.py
      - data/raw/house_data.csv
    outs:
      - data/processed/processed_house_data.csv

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/processed_house_data.csv
      - params.yaml
    outs:
      - models/model.pkl
```

## Why use `dvc.yaml`?

This defines the ML pipeline stages.

DVC now knows:

* what command to run
* what files each stage depends on
* what files each stage produces

This makes the project **reproducible**.

---

# 21. Step 17 — Run DVC Pipeline

## Command

```bash
dvc repro
```

## What does it do?

DVC checks dependencies and runs only the stages that need to run.

Examples:

* raw data changed → preprocess + train rerun
* only params changed → train rerun
* nothing changed → no unnecessary rerun

## Why is this powerful?

In real ML systems, retraining can be expensive.

DVC helps avoid waste and makes workflows smarter.

---

# 22. Step 18 — Save Everything in Git

## Commands

```bash
git add .
git commit -m "Initial MLOps house price project"
```

## Why commit now?

At this point you have:

* structured project
* working preprocessing
* working training
* saved model
* MLflow tracking
* FastAPI serving
* DVC pipeline
* code versioning with Git

This is your first complete MLOps milestone.

---

# 23. Mini Architecture Diagram (Cleanest Version)

```text
                ┌─────────────────────┐
                │   Raw CSV Dataset   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Preprocessing Script│
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Processed Dataset  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Training Script     │
                │ (Scikit-learn)      │
                └──────────┬──────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
┌─────────────────────┐            ┌─────────────────────┐
│ Save model.pkl      │            │ MLflow Tracking     │
└──────────┬──────────┘            └─────────────────────┘
           │
           ▼
┌─────────────────────┐
│ FastAPI Prediction  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ DVC Repro Pipeline  │
└─────────────────────┘
```

---

## Architecture explanation

* **Raw CSV Dataset** → original source data
* **Preprocessing Script** → cleans the data
* **Processed Dataset** → ready for ML training
* **Training Script** → creates the model
* **Save model.pkl** → stores trained artifact
* **MLflow Tracking** → stores params + metrics + artifacts
* **FastAPI Prediction** → serves predictions through API
* **DVC Repro Pipeline** → automates and reproduces the flow

---

# 24. What You Learned

By completing this project, you now understand:

* what a real ML project structure looks like
* how preprocessing works
* why train/test split matters
* how training works
* how inference works
* why model saving matters
* why MLflow matters
* why DVC matters
* why API serving matters
* how MLOps is more than just notebook training

---

# 25. Next Steps

Once this local version works, the next practical upgrade is:

## Part 3 (Recommended next)

* Add `Dockerfile`
* Run the API in Docker
* Fix `app.main:app` import path
* Deploy on **kind** using local Docker image
* Add `imagePullPolicy: Never`
* Add Kubernetes `Deployment` + `Service`
* Use `kubectl port-forward`
* Add `/metrics` endpoint for Prometheus
* Add GitHub Actions CI/CD later

---

# Final Beginner Summary

If you remember only one thing:

> **MLOps is the practice of making ML projects reproducible, trackable, and usable in real systems.**

This project showed you how to move from:

```text
CSV Data → Preprocess → Train → Save Model → Track with MLflow → Serve with FastAPI → Reproduce with DVC
```

That is the foundation of real-world MLOps.
