import os
from pathlib import Path

# Allow file store for MLflow tracking backend
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# User-provided raw data path
RAW_DATA_PATH = BASE_DIR / "creditcard.csv"

DATA_PATH = DATA_DIR / "transactions.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODEL_PATH = MODEL_DIR / "fraud_pipeline.pkl"
LOG_FILE = LOGS_DIR / "api.log"

# ML Pipeline Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# Columns
CATEGORICAL_FEATURES = [
    "merchant_category",
    "device_type"
]

# Numerical features include Time, Amount and PCA components V1 to V28
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
NUMERICAL_FEATURES = ["Time", "Amount"] + PCA_FEATURES

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "is_fraud"

# Model Hyperparameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "n_jobs": -1
}

# MLflow Configuration
MLFLOW_EXPERIMENT_NAME = "Fraud_Detection_System"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:///{BASE_DIR}/mlruns")

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
