import time
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from src.config import (
    TRAIN_PATH, TEST_PATH, MODEL_PATH,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET,
    MODEL_PARAMS, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
)
from src.utils import get_logger

logger = get_logger("pipeline_module")

def train_pipeline() -> None:
    """
    Trains the end-to-end ML model using ColumnTransformer preprocessing 
    and logs metrics, hyperparameters, and models to MLflow.
    """
    logger.info("Initializing MLflow experiment...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Load dataset splits
    logger.info(f"Loading training data from {TRAIN_PATH} and test data from {TEST_PATH}...")
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Training or testing splits not found. Please run the data preparation step first."
        )
        
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Separate features and target
    X_train = train_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y_test = test_df[TARGET]
    
    # Define ColumnTransformer Preprocessing (Imputation + Scaling/One-Hot)
    logger.info("Defining preprocessors...")
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'  # drop any other columns just in case
    )
    
    # Define full Pipeline (Preprocessing + Model Classifier)
    # This prevents any leakage because prep fit happens only during training pipeline fit
    clf = RandomForestClassifier(**MODEL_PARAMS)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    # Run training inside MLflow
    logger.info("Starting MLflow run...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        # Log params
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("num_features", len(NUMERICAL_FEATURES))
        mlflow.log_param("cat_features", len(CATEGORICAL_FEATURES))
        
        # Train model
        logger.info("Fitting end-to-end pipeline model on train split...")
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        duration = time.time() - start_time
        logger.info(f"Model training finished in {duration:.2f} seconds.")
        
        # Log training duration
        mlflow.log_metric("training_duration_seconds", duration)
        
        # Evaluate metrics on Test Split
        logger.info("Evaluating model on test split...")
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "precision_recall_auc": average_precision_score(y_test, y_prob)
        }
        
        # Print metrics
        logger.info("Test Set Evaluation Metrics:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.6f}")
            mlflow.log_metric(name, value)
            
        # Log model artifact to MLflow
        logger.info("Logging model artifact to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="FraudDetectionPipeline"
        )
        
        # Serialize model locally as a fallback
        logger.info(f"Saving model pipeline locally to {MODEL_PATH}...")
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(pipeline, f)
            
        logger.info("ML pipeline training run complete successfully.")

if __name__ == "__main__":
    train_pipeline()
