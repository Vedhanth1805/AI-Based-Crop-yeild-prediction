"""
Prediction Module
=================
Load saved models and make predictions with confidence/prediction intervals.
Covers: Unit 3 (prediction intervals), Unit 5 (model inference).

Author: B.Tech AI & Data Science Student
Course: ISPA (21AIC401T)
"""

import pandas as pd
import numpy as np
import os
import joblib
import statsmodels.api as sm

from src.data_preprocessing import FEATURE_DISPLAY, FEATURE_NAMES

# Model directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_models():
    """
    Load all saved models, pipeline, and metadata.
    
    Returns
    -------
    dict
        Dictionary containing models, pipeline, feature info.
    
    Raises
    ------
    FileNotFoundError
        If model files are missing (run training first).
    """
    required_files = {
        "linear_regression": os.path.join(MODEL_DIR, "linear_regression.joblib"),
        "cart_tree": os.path.join(MODEL_DIR, "cart_tree.joblib"),
        "random_forest": os.path.join(MODEL_DIR, "random_forest.joblib"),
        "pipeline": os.path.join(MODEL_DIR, "preprocessing_pipeline.joblib"),
        "feature_columns": os.path.join(MODEL_DIR, "feature_columns.joblib"),
        "feature_ranges": os.path.join(MODEL_DIR, "feature_ranges.joblib"),
    }
    
    missing = [name for name, path in required_files.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}. "
            f"Run 'python -m src.model_training' first to train models."
        )
    
    models = {}
    for name, path in required_files.items():
        models[name] = joblib.load(path)
    
    # Load OLS model if available (for prediction intervals)
    ols_path = os.path.join(MODEL_DIR, "ols_statsmodels.joblib")
    if os.path.exists(ols_path):
        models["ols_statsmodels"] = joblib.load(ols_path)
    
    return models


def predict(feature_values, models=None):
    """
    Make predictions using all trained models.
    
    Parameters
    ----------
    feature_values : dict
        {feature_name: value} e.g. {"rainfall": 650, "temperature": 28, ...}
    models : dict, optional
        Pre-loaded models. If None, loads from disk.
    
    Returns
    -------
    dict
        Predictions from each model, prediction intervals, out-of-range warnings.
    """
    if models is None:
        models = load_models()
    
    feature_cols = models["feature_columns"]
    pipeline = models["pipeline"]
    feature_ranges = models["feature_ranges"]
    
    # Create input DataFrame in correct column order
    input_data = {}
    warnings_list = []
    for col in feature_cols:
        if col in feature_values:
            val = float(feature_values[col])
            input_data[col] = [val]
            
            # Check if input is outside training range
            if col in feature_ranges:
                fmin, fmax = feature_ranges[col]
                if val < fmin or val > fmax:
                    display = FEATURE_DISPLAY.get(col, col)
                    warnings_list.append(
                        f"⚠️ {display} = {val:.1f} is outside the training range "
                        f"[{fmin:.1f}, {fmax:.1f}]. Prediction may be unreliable."
                    )
        else:
            raise ValueError(f"Missing feature: {col}")
    
    X_raw = pd.DataFrame(input_data)
    
    # Preprocess using the saved pipeline
    X_processed = pd.DataFrame(
        pipeline.transform(X_raw),
        columns=feature_cols
    )
    
    # Predictions from each model
    results = {
        "warnings": warnings_list,
        "predictions": {},
    }
    
    # Linear Regression
    lr_pred = float(models["linear_regression"].predict(X_processed)[0])
    results["predictions"]["Linear Regression"] = round(lr_pred, 2)
    
    # CART Tree
    cart_pred = float(models["cart_tree"].predict(X_processed)[0])
    results["predictions"]["CART Tree"] = round(cart_pred, 2)
    
    # Random Forest
    rf_pred = float(models["random_forest"].predict(X_processed)[0])
    results["predictions"]["Random Forest (Extension)"] = round(rf_pred, 2)
    
    # Prediction interval from OLS (on raw data, Unit 3)
    if "ols_statsmodels" in models:
        try:
            ols_model = models["ols_statsmodels"]
            X_raw_const = sm.add_constant(X_raw, has_constant="add")
            prediction = ols_model.get_prediction(X_raw_const)
            pred_frame = prediction.summary_frame(alpha=0.05)
            results["prediction_interval"] = {
                "point_estimate": round(float(pred_frame["mean"].iloc[0]), 2),
                "ci_lower": round(float(pred_frame["mean_ci_lower"].iloc[0]), 2),
                "ci_upper": round(float(pred_frame["mean_ci_upper"].iloc[0]), 2),
                "pi_lower": round(float(pred_frame["obs_ci_lower"].iloc[0]), 2),
                "pi_upper": round(float(pred_frame["obs_ci_upper"].iloc[0]), 2),
            }
        except Exception as e:
            results["prediction_interval"] = {"error": str(e)}
    
    return results


def check_input_ranges(feature_values, feature_ranges):
    """
    Check if input values are within the training data range.
    
    Parameters
    ----------
    feature_values : dict
        Input feature values.
    feature_ranges : dict
        {feature: (min, max)} from training data.
    
    Returns
    -------
    list
        Warning messages for out-of-range inputs.
    """
    warnings_list = []
    for col, val in feature_values.items():
        if col in feature_ranges:
            fmin, fmax = feature_ranges[col]
            if val < fmin or val > fmax:
                display = FEATURE_DISPLAY.get(col, col)
                warnings_list.append(
                    f"⚠️ {display} = {val:.1f} is outside training range "
                    f"[{fmin:.1f}, {fmax:.1f}]"
                )
    return warnings_list


if __name__ == "__main__":
    # Quick test with sample values
    test_input = {
        "rainfall": 1500,
        "temperature": 25,
        "soil_ph": 6.5,
        "fertilizer": 200,
    }
    
    try:
        result = predict(test_input)
        print("Predictions:")
        for model, pred in result["predictions"].items():
            print(f"  {model}: {pred:.2f} kg/ha")
        if "prediction_interval" in result:
            pi = result["prediction_interval"]
            print(f"\nRegression 95% Prediction Interval:")
            print(f"  [{pi.get('pi_lower', 'N/A')}, {pi.get('pi_upper', 'N/A')}] kg/ha")
        if result["warnings"]:
            print("\nWarnings:")
            for w in result["warnings"]:
                print(f"  {w}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
