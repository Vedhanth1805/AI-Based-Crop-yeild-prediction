"""
Data Preprocessing Module
=========================
Handles loading, cleaning, validating and splitting the crop yield dataset.
Covers: Unit 4 (data partitioning, leakage prevention).

Author: B.Tech AI & Data Science Student
Course: ISPA (21AIC401T)
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================================
# COLUMN MAPPING DICTIONARY
# ============================================================================
# Change these mappings if your CSV has different column names.
# Keys = internal names used throughout the project
# Values = actual column names in your CSV file
COLUMN_MAP = {
    "rainfall": "Rainfall",           # Rainfall in mm
    "temperature": "Temperature",     # Temperature in °C
    "soil_ph": "Soil_pH",             # Soil pH value
    "fertilizer": "Fertilizer_Used",  # Fertilizer amount in kg/hectare
    "yield": "Crop_Yield_ton_per_hectare",  # Crop yield (will be converted to kg/ha)
    "soil_type": "Soil_Type",         # Categorical: Clay, Sandy, Silt, Loamy
    "crop_type": "Crop_Type",         # Categorical: Rice, Wheat, etc.
}

# The four numeric features we use for prediction
FEATURE_NAMES = ["rainfall", "temperature", "soil_ph", "fertilizer"]

# Display-friendly names and units
FEATURE_DISPLAY = {
    "rainfall": "Rainfall (mm)",
    "temperature": "Temperature (°C)",
    "soil_ph": "Soil pH",
    "fertilizer": "Fertilizer (kg/ha)",
}

TARGET_DISPLAY = "Crop Yield (kg/ha)"

# Whether this is real or simulated data
DATA_SOURCE = "REAL"  # Set to "SIMULATED" if using synthetic fallback

# Crop type filter: The original dataset has 6 crop types with vastly different
# yield ranges (sugarcane ~75 t/ha vs wheat ~8 t/ha). Crop type dominates yield,
# making the 4 numeric features appear uncorrelated when all crops are mixed.
# Filtering to one crop type reveals the real feature-yield relationships.
# Set to None to use all crops (will need one-hot encoding of crop type).
CROP_FILTER = "Rice"  # Filter to Rice (n≈1700) for meaningful analysis

# Fixed random state for reproducibility (Unit 4)
RANDOM_STATE = 42

# Data file path
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "crop_data.csv")


def load_and_validate_data(filepath=None):
    """
    Load the CSV dataset and validate that required columns exist.
    Converts yield from ton/hectare to kg/hectare.
    
    Unit 4: Data loading and validation.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to DATA_FILE.
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardized column names.
    
    Raises
    ------
    FileNotFoundError
        If the data file doesn't exist.
    ValueError
        If required columns are missing.
    """
    if filepath is None:
        filepath = DATA_FILE
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            f"Please place your CSV file at data/crop_data.csv"
        )
    
    df = pd.read_csv(filepath)
    
    # Check that all required columns exist
    missing_cols = []
    for internal_name, csv_col in COLUMN_MAP.items():
        if csv_col not in df.columns:
            missing_cols.append(f"'{csv_col}' (for {internal_name})")
    
    if missing_cols:
        raise ValueError(
            f"Missing columns in dataset: {', '.join(missing_cols)}. "
            f"Available columns: {list(df.columns)}. "
            f"Update COLUMN_MAP in data_preprocessing.py to match your CSV."
        )
    
    # Rename columns to internal names for consistency
    reverse_map = {v: k for k, v in COLUMN_MAP.items() if v in df.columns}
    df = df.rename(columns=reverse_map)
    
    # Filter by crop type if configured (to avoid crop-type variance masking environmental factor effects)
    if CROP_FILTER and "crop_type" in df.columns:
        df = df[df["crop_type"] == CROP_FILTER].copy().reset_index(drop=True)
        print(f"✓ Filtered to Crop_Type == '{CROP_FILTER}' ({len(df)} records)")
    
    # Convert yield from ton/hectare to kg/hectare
    if "yield" in df.columns:
        df["yield"] = df["yield"] * 1000.0
    
    print(f"✓ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Source: {filepath}")
    print(f"  Data type: {DATA_SOURCE}")
    
    return df


def data_quality_report(df):
    """
    Generate a comprehensive data quality report.
    
    Unit 4: Data understanding and quality assessment.
    
    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.
    
    Returns
    -------
    dict
        Dictionary containing quality metrics.
    """
    feature_cols = [f for f in FEATURE_NAMES if f in df.columns] + ["yield"]
    df_subset = df[feature_cols]
    
    report = {
        "shape": df.shape,
        "dtypes": df_subset.dtypes.to_dict(),
        "missing_values": df_subset.isnull().sum().to_dict(),
        "missing_pct": (df_subset.isnull().sum() / len(df) * 100).to_dict(),
        "duplicates": df.duplicated().sum(),
        "descriptive_stats": df_subset.describe().round(2),
    }
    
    # IQR-based outlier detection
    outliers = {}
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            outliers[col] = {
                "Q1": round(Q1, 2), "Q3": round(Q3, 2), "IQR": round(IQR, 2),
                "lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
                "n_outliers": int(n_outliers),
                "pct_outliers": round(n_outliers / len(df) * 100, 2),
            }
    report["outliers"] = outliers
    
    return report


def split_data(df, train_size=0.70, val_size=0.15, test_size=0.15):
    """
    Split data into train / validation / test sets (70/15/15).
    
    Unit 4: Data partitioning and leakage prevention.
    The split is done BEFORE any imputation or scaling to prevent data leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with features and target.
    train_size : float
        Proportion for training (default 0.70).
    val_size : float
        Proportion for validation (default 0.15).
    test_size : float
        Proportion for testing (default 0.15).
    
    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9, \
        "Split proportions must sum to 1.0"
    
    feature_cols = [f for f in FEATURE_NAMES if f in df.columns]
    X = df[feature_cols].copy()
    y = df["yield"].copy()
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size),
        random_state=RANDOM_STATE
    )
    
    # Second split: val vs test (split the remainder equally)
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio),
        random_state=RANDOM_STATE
    )
    
    print(f"✓ Data split (before preprocessing):")
    print(f"  Train: {X_train.shape[0]} samples ({train_size*100:.0f}%)")
    print(f"  Validation: {X_val.shape[0]} samples ({val_size*100:.0f}%)")
    print(f"  Test: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_preprocessing_pipeline():
    """
    Create an sklearn Pipeline for preprocessing.
    
    Unit 4: Data leakage prevention using pipelines.
    - Median imputation: robust to outliers (fitted on training set only).
    - StandardScaler: needed for regression coefficient comparability.
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Preprocessing pipeline.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return pipeline


def preprocess_data(X_train, X_val, X_test):
    """
    Fit preprocessing on training data and transform all sets.
    
    Unit 4: The pipeline is fit ONLY on training data to prevent leakage.
    
    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature DataFrames.
    
    Returns
    -------
    tuple
        (X_train_processed, X_val_processed, X_test_processed, pipeline)
    """
    feature_cols = X_train.columns.tolist()
    pipeline = create_preprocessing_pipeline()
    
    # Fit on training data ONLY (Unit 4: leakage prevention)
    X_train_processed = pd.DataFrame(
        pipeline.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    
    # Transform validation and test using training statistics
    X_val_processed = pd.DataFrame(
        pipeline.transform(X_val),
        columns=feature_cols,
        index=X_val.index
    )
    X_test_processed = pd.DataFrame(
        pipeline.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )
    
    print(f"✓ Preprocessing complete (fitted on training data only)")
    print(f"  Imputation strategy: median")
    print(f"  Scaling: StandardScaler (mean=0, std=1)")
    
    return X_train_processed, X_val_processed, X_test_processed, pipeline


def get_available_features(df):
    """
    Return list of features that actually exist in the dataset.
    
    Returns
    -------
    list
        List of available feature internal names.
    """
    return [f for f in FEATURE_NAMES if f in df.columns]


def get_feature_ranges(X_train_raw):
    """
    Get the min/max range of each feature in the raw training data.
    Used for out-of-range warnings in the prediction page.
    
    Parameters
    ----------
    X_train_raw : pd.DataFrame
        Raw (unscaled) training features.
    
    Returns
    -------
    dict
        {feature_name: (min, max)}
    """
    ranges = {}
    for col in X_train_raw.columns:
        ranges[col] = (float(X_train_raw[col].min()), float(X_train_raw[col].max()))
    return ranges


if __name__ == "__main__":
    # Quick test
    df = load_and_validate_data()
    report = data_quality_report(df)
    print("\n--- Data Quality Report ---")
    print(f"Shape: {report['shape']}")
    print(f"Missing: {report['missing_values']}")
    print(f"Duplicates: {report['duplicates']}")
    print(f"\nDescriptive Statistics:")
    print(report['descriptive_stats'])
    print(f"\nOutliers (IQR method):")
    for col, info in report['outliers'].items():
        print(f"  {col}: {info['n_outliers']} ({info['pct_outliers']}%)")
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_p, X_val_p, X_test_p, pipe = preprocess_data(X_train, X_val, X_test)
