# Fraud Detection System — End-to-End ML Pipeline

This project implements a production-grade, end-to-end fraud detection machine learning pipeline based on the Kaggle Credit Card Fraud Detection dataset (`creditcard.csv`).

## Features

- **Preprocessing & Leakage Control**: Employs scikit-learn `ColumnTransformer` (median categorical/numerical imputation, standard scaling, and one-hot encoding) bundled inside a single prediction pipeline to prevent leakage between splits.
- **Stratified Data Splits**: Data splitter maintains label ratios (0.17% fraud rate) across train, validation, and test datasets.
- **Rest prediction serving**: A **FastAPI** web service serving predictions, featuring structured JSON logging (latency, request metadata) and **Pydantic** request schema validation.
- **Monitoring & Threshold Evaluation**: A **Streamlit** dashboard visualizing ROC and Precision-Recall Curves, confusion matrices, and interactive threshold tuning with an asymmetric financial cost optimizer.
- **Reproducibility**: Experiments, hyperparameters, and models are versioned and logged using **MLflow**.
- **Orchestration**: Easy setup and running via Docker Compose, Makefile, and Windows native PowerShell (`run.ps1`) scripts.

---

## Directory Structure

```text
fraud-detection/
├── src/
│   ├── __init__.py
│   ├── config.py         # Configs, column mapping, and hyperparameters
│   ├── data.py           # Preprocessing, categorical enrichment, and splits
│   ├── pipeline.py       # Preprocessing & RandomForest classifier + MLflow logging
│   ├── api.py            # FastAPI predictive REST service with JSON logging middleware
│   ├── app.py            # Streamlit dashboard (ROC, PR, Threshold/Cost tuning)
│   └── utils.py          # Structured JSON logging helpers
├── tests/
│   ├── __init__.py
│   ├── test_data.py      # Test stratified splitting and leakage logic
│   ├── test_pipeline.py  # Test ColumnTransformer preprocessing & imputation
│   └── test_api.py       # Test API health and prediction endpoints
├── Dockerfile            # Container definition for serving API
├── docker-compose.yml    # Multi-container orchestration (FastAPI + Streamlit)
├── Makefile              # Orchestration commands for Unix systems
├── run.ps1               # Orchestration helper for Windows (PowerShell)
├── requirements.txt      # Python dependencies
└── README.md             # This document
```

---

## Quick Start (Local Run)

### 1. Setup Environment
Install dependencies in a virtual environment:
```powershell
# Windows
.\run.ps1 setup

# Unix
make setup
```

### 2. Preprocess Data
Enrich `creditcard.csv` with missing values/categories and generate stratified splits:
```powershell
# Windows
.\run.ps1 data

# Unix
make data
```

### 3. Run Pipeline Training
Fit the model pipeline, log parameters/metrics to MLflow, and serialize:
```powershell
# Windows
.\run.ps1 train

# Unix
make train
```

### 4. Serve Predictions
Run the FastAPI prediction server locally on port 8000:
```powershell
# Windows
.\run.ps1 serve

# Unix
make serve
```
- Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Uptime Check: [http://localhost:8000/health](http://localhost:8000/health)

### 5. Launch Evaluation Dashboard
Run the Streamlit interactive dashboard:
```powershell
# Windows
.\run.ps1 dashboard

# Unix
make dashboard
```

---

## Running in Docker Containers

To run both services in containerized environments:

```powershell
# Build containers
.\run.ps1 docker-build  # or 'make docker-build'

# Run containers
.\run.ps1 docker-run    # or 'make docker-run'
```
- **REST Serving API**: `http://localhost:8000`
- **Streamlit Evaluation Dashboard**: `http://localhost:8501`

---

## Running Tests
Run the automated test suite with pytest:
```powershell
# Windows
.\run.ps1 test

# Unix
make test
```
