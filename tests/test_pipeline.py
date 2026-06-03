import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.pipeline import train_pipeline
from src.data import generate_synthetic_fallback
from src.config import TARGET

def test_pipeline_training(tmp_path, monkeypatch):
    """
    Verifies that the end-to-end model training runs successfully, 
    preprocesses columns, and serializes a functional pipeline.
    """
    # Create tiny mock datasets for training
    train_df = generate_synthetic_fallback(num_samples=200, fraud_rate=0.05)
    test_df = generate_synthetic_fallback(num_samples=50, fraud_rate=0.05)
    
    # Paths in config to mock
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    model_file = tmp_path / "fraud_pipeline.pkl"
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    monkeypatch.setattr("src.pipeline.TRAIN_PATH", train_file)
    monkeypatch.setattr("src.pipeline.TEST_PATH", test_file)
    monkeypatch.setattr("src.pipeline.MODEL_PATH", model_file)
    
    # Disable actual MLflow remote logging for testing to avoid connection issues,
    # or just let it write to local directory since tracking URI is configured locally.
    monkeypatch.setattr("src.pipeline.MLFLOW_TRACKING_URI", f"file:///{tmp_path}/mlruns")
    
    # Train
    train_pipeline()
    
    # Assert model file was written
    assert model_file.exists()
    
    # Load and test prediction capabilities
    with open(model_file, "rb") as f:
        loaded_pipeline = pickle.load(f)
        
    assert isinstance(loaded_pipeline, Pipeline)
    
    # Create a prediction payload containing missing values
    payload = pd.DataFrame([{
        "merchant_category": "electronics",
        "device_type": None,  # Missing categorical
        "Time": 1000.0,
        "Amount": np.nan,    # Missing numerical
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }])
    
    # Inference on input with NaNs
    pred_prob = loaded_pipeline.predict_proba(payload)[0]
    
    # Assert output probability is calculated correctly
    assert len(pred_prob) == 2
    assert 0.0 <= pred_prob[0] <= 1.0
    assert 0.0 <= pred_prob[1] <= 1.0
