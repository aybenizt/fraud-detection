import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.api import app, TransactionRequest
import src.api

class MockPipeline:
    """
    Mock class mimicking the trained scikit-learn pipeline object for API tests.
    """
    def predict_proba(self, X):
        # Mock class returns a probability based on transaction amount
        # Let's say high amounts (e.g. > 500) have high fraud probability
        N = len(X)
        probs = np.zeros((N, 2))
        amounts = X["Amount"].values
        
        for idx, amt in enumerate(amounts):
            if amt > 500:
                probs[idx, 0] = 0.2
                probs[idx, 1] = 0.8
            else:
                probs[idx, 0] = 0.98
                probs[idx, 1] = 0.02
        return probs

@pytest.fixture(autouse=True)
def patch_model_pipeline(monkeypatch):
    """
    Automatically patch the model pipeline inside the api module for tests.
    """
    monkeypatch.setattr(src.api, "model_pipeline", MockPipeline())

client = TestClient(app)

def test_health_endpoint():
    """
    Verifies that the GET /health healthcheck returns a successful status.
    """
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["model_loaded"] == "True"

def test_predict_legitimate_transaction():
    """
    Verifies prediction response schema and value for a normal transaction.
    """
    payload = {
        "merchant_category": "groceries",
        "device_type": "desktop",
        "Time": 100.0,
        "Amount": 45.50,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }
    
    response = client.post("/predict?threshold=0.5", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "decision_threshold" in data
    assert "latency_ms" in data
    
    assert data["fraud_probability"] == 0.02
    assert data["is_fraud"] == 0
    assert data["decision_threshold"] == 0.5

def test_predict_fraudulent_transaction():
    """
    Verifies prediction categorizes high amount transaction as fraud under threshold.
    """
    payload = {
        "merchant_category": "travel",
        "device_type": "mobile",
        "Time": 200.0,
        "Amount": 999.0,  # High amount triggers fraud in mock
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }
    
    response = client.post("/predict?threshold=0.5", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["fraud_probability"] == 0.8
    assert data["is_fraud"] == 1

def test_predict_invalid_schema():
    """
    Verifies that invalid request payloads return 422 validation errors.
    """
    # Missing V1-V28 PCA values (but our request fields have defaults, let's omit Amount which is required)
    # Wait, in TransactionRequest, we set default values for all V1..V28 and Amount=50.0. 
    # Let's send an invalid type (Amount as string that cannot be parsed as float) to trigger validation error.
    payload = {
        "Amount": "invalid-amount-string",
        "merchant_category": "groceries"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
