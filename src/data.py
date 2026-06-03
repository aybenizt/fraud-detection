import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import (
    RAW_DATA_PATH, DATA_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH,
    RANDOM_STATE, TEST_SIZE, VAL_SIZE, TARGET
)
from src.utils import get_logger

logger = get_logger("data_module")

def load_and_enrich_data() -> pd.DataFrame:
    """
    Loads raw creditcard.csv data and enriches it with categorical columns 
    and missing values to test ColumnTransformer preprocessing.
    """
    if not RAW_DATA_PATH.exists():
        logger.warning(f"Raw data file not found at {RAW_DATA_PATH}. Generating synthetic fallback data...")
        return generate_synthetic_fallback()
        
    logger.info(f"Loading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Rename target
    if "Class" in df.columns:
        df = df.rename(columns={"Class": TARGET})
        
    num_samples = len(df)
    logger.info(f"Loaded {num_samples} rows. Enriching with categorical columns and introducing missing values...")
    
    # Set seed for reproducibility
    np.random.seed(RANDOM_STATE)
    
    # Inject synthetic categorical features to meet ColumnTransformer (imputation + OHE) requirement
    merchant_categories = ["groceries", "electronics", "travel", "entertainment", "services"]
    # We make travel and electronics slightly more common in fraud
    cat_weights_normal = [0.4, 0.2, 0.1, 0.15, 0.15]
    cat_weights_fraud = [0.1, 0.4, 0.3, 0.1, 0.1]
    
    merchant_cat = np.empty(num_samples, dtype=object)
    fraud_mask = df[TARGET] == 1
    
    merchant_cat[~fraud_mask] = np.random.choice(merchant_categories, size=(~fraud_mask).sum(), p=cat_weights_normal)
    merchant_cat[fraud_mask] = np.random.choice(merchant_categories, size=fraud_mask.sum(), p=cat_weights_fraud)
    df["merchant_category"] = merchant_cat
    
    device_types = ["mobile", "desktop", "tablet"]
    device_weights_normal = [0.5, 0.4, 0.1]
    device_weights_fraud = [0.8, 0.15, 0.05]  # mobile is highly prone to fraud
    
    device = np.empty(num_samples, dtype=object)
    device[~fraud_mask] = np.random.choice(device_types, size=(~fraud_mask).sum(), p=device_weights_normal)
    device[fraud_mask] = np.random.choice(device_types, size=fraud_mask.sum(), p=device_weights_fraud)
    df["device_type"] = device
    
    # Inject missing values (NaNs) to test Imputer (2% in Amount, 1% in V1, 2% in device_type)
    amount_nan_mask = np.random.rand(num_samples) < 0.02
    v1_nan_mask = np.random.rand(num_samples) < 0.01
    device_nan_mask = np.random.rand(num_samples) < 0.02
    
    df.loc[amount_nan_mask, "Amount"] = np.nan
    df.loc[v1_nan_mask, "V1"] = np.nan
    df.loc[device_nan_mask, "device_type"] = None
    
    actual_fraud_rate = df[TARGET].mean()
    logger.info(f"Data enrichment complete. Fraud rate: {actual_fraud_rate:.4%} ({df[TARGET].sum()} positive cases).")
    return df

def generate_synthetic_fallback(num_samples: int = 50000, fraud_rate: float = 0.015) -> pd.DataFrame:
    """
    Generates a synthetic fallback dataset mimicking the Credit Card Fraud structure.
    """
    logger.info(f"Generating {num_samples} synthetic transaction rows...")
    np.random.seed(RANDOM_STATE)
    
    # Generate target labels
    is_fraud = np.random.choice([0, 1], size=num_samples, p=[1 - fraud_rate, fraud_rate])
    
    # Generate Time and Amount
    time = np.random.uniform(0, 172800, size=num_samples)
    amount = np.random.lognormal(mean=3.5, sigma=1.2, size=num_samples)
    amount = np.round(amount, 2)
    
    # Generate V1 to V28
    pca_data = {}
    for i in range(1, 29):
        pca_data[f"V{i}"] = np.random.normal(loc=0, scale=1.5 if i <= 5 else 0.8, size=num_samples)
        
    df = pd.DataFrame(pca_data)
    df["Time"] = time
    df["Amount"] = amount
    df[TARGET] = is_fraud
    
    # Enrich with categoricals
    merchant_categories = ["groceries", "electronics", "travel", "entertainment", "services"]
    df["merchant_category"] = np.random.choice(merchant_categories, size=num_samples)
    
    device_types = ["mobile", "desktop", "tablet"]
    df["device_type"] = np.random.choice(device_types, size=num_samples)
    
    # Inject missing values
    df.loc[np.random.rand(num_samples) < 0.02, "Amount"] = np.nan
    df.loc[np.random.rand(num_samples) < 0.02, "device_type"] = None
    
    return df

def save_and_split_data(df: pd.DataFrame) -> None:
    """
    Performs stratified train/val/test splits to control leakage and keep label ratio.
    """
    logger.info("Splitting data into train/val/test sets using stratified splits...")
    
    # Save the full dataset
    df.to_csv(DATA_PATH, index=False)
    
    # Split into Train and Temp (val + test)
    # Stratified split based on target label
    train_df, temp_df = train_test_split(
        df,
        test_size=(TEST_SIZE + VAL_SIZE),
        stratify=df[TARGET],
        random_state=RANDOM_STATE
    )
    
    # Split Temp into Val and Test (evenly based on original sizes)
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio),
        stratify=temp_df[TARGET],
        random_state=RANDOM_STATE
    )
    
    # Save splits
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    
    logger.info(f"Data split and saved: "
                f"Train size={len(train_df)} (fraud={train_df[TARGET].sum()}), "
                f"Val size={len(val_df)} (fraud={val_df[TARGET].sum()}), "
                f"Test size={len(test_df)} (fraud={test_df[TARGET].sum()})")

if __name__ == "__main__":
    df = load_and_enrich_data()
    save_and_split_data(df)
