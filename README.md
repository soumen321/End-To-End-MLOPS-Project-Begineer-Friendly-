
### Project structure

mlops-house-price/
│
├── data/
│   ├── raw/
│   │   └── house_data.csv
│   └── processed/
│
├── models/
│   └── model.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── app/
│   └── main.py
│
├── notebooks/
│
├── tests/
│
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md

## Why this folder structure?

- data/raw/

stores original dataset
Why use it?
Never change original data directly. Keep raw data safe.

- data/processed/

stores cleaned/transformed data

Why use it?
You separate original data from cleaned data.

models/

stores trained model file (model.pkl)

Why use it?
Model is an artifact. It should be stored separately.

src/

all ML pipeline code

Why use it?
Keeps training logic organized and production-friendly.

app/

API code (FastAPI)

Why use it?
Training code and serving code should be separate.

dvc.yaml

pipeline steps for DVC

Why use it?
Automates and tracks the ML pipeline.

params.yaml

hyperparameters config file

Why use it?
Instead of hardcoding values in Python, keep them configurable.


