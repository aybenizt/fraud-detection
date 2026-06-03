import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import pickle
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.config import (
    MODEL_PATH, FEATURES, TARGET, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
)
from src.utils import get_logger


# Initialize logger and FastAPI
logger = get_logger("api_service")
app = FastAPI(
    title="Fraud Detection API",
    description="Production-grade REST service for credit card fraud prediction.",
    version="1.0.0"
)

# Load the trained model pipeline global variable
model_pipeline = None

try:
    if MODEL_PATH.exists():
        logger.info(f"Loading trained model pipeline from {MODEL_PATH}...")
        with open(MODEL_PATH, "rb") as f:
            model_pipeline = pickle.load(f)
        logger.info("Model pipeline loaded successfully.")
    else:
        logger.warning(f"Model pipeline not found at {MODEL_PATH}. Prediction service will return 503 until trained.")
except Exception as e:
    logger.exception(f"Error loading model pipeline: {e}")

# Define Pydantic request body schema dynamically supporting V1-V28
class TransactionRequest(BaseModel):
    # Categorical features
    merchant_category: Optional[str] = Field(default="groceries", description="Merchant category of transaction")
    device_type: Optional[str] = Field(default="mobile", description="Device type used for transaction")
    
    # Core Numerical features
    Time: float = Field(default=0.0, description="Seconds elapsed since the first transaction")
    Amount: float = Field(default=50.0, description="Transaction amount")
    
    # PCA components V1 to V28
    V1: float = Field(default=0.0)
    V2: float = Field(default=0.0)
    V3: float = Field(default=0.0)
    V4: float = Field(default=0.0)
    V5: float = Field(default=0.0)
    V6: float = Field(default=0.0)
    V7: float = Field(default=0.0)
    V8: float = Field(default=0.0)
    V9: float = Field(default=0.0)
    V10: float = Field(default=0.0)
    V11: float = Field(default=0.0)
    V12: float = Field(default=0.0)
    V13: float = Field(default=0.0)
    V14: float = Field(default=0.0)
    V15: float = Field(default=0.0)
    V16: float = Field(default=0.0)
    V17: float = Field(default=0.0)
    V18: float = Field(default=0.0)
    V19: float = Field(default=0.0)
    V20: float = Field(default=0.0)
    V21: float = Field(default=0.0)
    V22: float = Field(default=0.0)
    V23: float = Field(default=0.0)
    V24: float = Field(default=0.0)
    V25: float = Field(default=0.0)
    V26: float = Field(default=0.0)
    V27: float = Field(default=0.0)
    V28: float = Field(default=0.0)

    class Config:
        json_schema_extra = {
            "example": {
                "merchant_category": "electronics",
                "device_type": "mobile",
                "Time": 86400.0,
                "Amount": 999.99,
                "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37,
                "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09,
                "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61,
                "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
                "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
                "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06,
                "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02
            }
        }

class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="Predicted probability of the transaction being fraudulent")
    is_fraud: int = Field(..., description="Binary prediction: 1 if fraudulent, 0 otherwise")
    decision_threshold: float = Field(..., description="The decision threshold used for classification")
    latency_ms: float = Field(..., description="Latency of prediction processing in milliseconds")

# Structured Request/Response Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    
    extra_fields = {
        "method": request.method,
        "url": str(request.url.path),
        "status_code": response.status_code,
        "duration_ms": round(process_time, 2)
    }
    
    logger.info(
        f"API HTTP request processed: {request.method} {request.url.path} -> {response.status_code}",
        extra={"extra_fields": extra_fields}
    )
    
    return response

@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Health check endpoint to monitor API service availability.
    """
    status = "healthy" if model_pipeline is not None else "degraded (model not loaded)"
    return {"status": status, "model_loaded": str(model_pipeline is not None)}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionRequest, threshold: float = 0.5) -> PredictionResponse:
    """
    Predicts the likelihood of fraud for a single transaction.
    """
    global model_pipeline
    
    # Load model on-demand if it wasn't loaded at startup (e.g. trained after service started)
    if model_pipeline is None:
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    model_pipeline = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load model pipeline: {e}")
        
        if model_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Prediction model is not available. Please train the model pipeline first."
            )
            
    start_time = time.time()
    
    try:
        # Convert request data to DataFrame matching model features in correct order
        data_dict = payload.model_dump()
        input_df = pd.DataFrame([data_dict])
        
        # Select features in the exact configured order to avoid transformer alignment issues
        input_df = input_df[FEATURES]
        
        # Predict probability
        # pipeline output shape is [1, 2], we take class 1 probability
        probs = model_pipeline.predict_proba(input_df)[0]
        fraud_prob = float(probs[1])
        
        # Apply prediction threshold
        is_fraud = 1 if fraud_prob >= threshold else 0
        
        latency = (time.time() - start_time) * 1000
        
        # Log prediction result structurally
        logger.info(
            f"Prediction completed. Fraud Probability: {fraud_prob:.4f} -> Target: {is_fraud}",
            extra={
                "extra_fields": {
                    "amount": payload.Amount,
                    "merchant": payload.merchant_category,
                    "device": payload.device_type,
                    "probability": fraud_prob,
                    "prediction": is_fraud,
                    "threshold": threshold,
                    "latency_ms": latency
                }
            }
        )
        
        return PredictionResponse(
            fraud_probability=round(fraud_prob, 5),
            is_fraud=is_fraud,
            decision_threshold=threshold,
            latency_ms=round(latency, 2)
        )
        
    except Exception as e:
        logger.exception(f"Error occurred during inference: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference processing error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    uvicorn.run("src.api:app", host=API_HOST, port=API_PORT, reload=True)
