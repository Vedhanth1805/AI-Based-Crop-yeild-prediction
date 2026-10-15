"""
Evaluation Module
=================
Model evaluation metrics, comparison tables, and final test evaluation.
Covers: Unit 5 (model evaluation and comparison).

Author: B.Tech AI & Data Science Student
Course: ISPA (21AIC401T)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_all_metrics(y_true, y_pred):
    """
    Compute comprehensive evaluation metrics.
    
    Unit 5: Model evaluation.
    
    Formulas:
    - MAE  = (1/n) Σ |yi - ŷi|        → Average absolute error
    - MSE  = (1/n) Σ (yi - ŷi)²       → Average squared error (penalises large errors)
    - RMSE = √MSE                       → Same unit as target, penalises large errors
    - R²   = 1 - SS_res / SS_tot       → Proportion of variance explained
    
    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    
    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": round(float(mae), 2),
        "MSE": round(float(mse), 2),
        "RMSE": round(float(rmse), 2),
        "R²": round(float(r2), 4),
    }


def create_comparison_table(results_dict):
    """
    Create a formatted model comparison table.
    
    Unit 5: Model comparison.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary from training results containing model metrics.
    
    Returns
    -------
    pd.DataFrame
        Comparison table with models as rows and metrics as columns.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        row = {"Model": model_name}
        for key in ["train_MAE", "train_RMSE", "train_R2",
                     "val_MAE", "val_RMSE", "val_R2",
                     "cv_mae"]:
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.columns = [c.replace("train_", "Train ").replace("val_", "Val ").replace("cv_", "CV ") for c in df.columns]
    return df


def format_test_results(test_results):
    """
    Format final test set results as a clean table.
    
    Unit 5: Final holdout evaluation.
    
    Parameters
    ----------
    test_results : dict
        {model_name: {test_MAE, test_MSE, test_RMSE, test_R2}}
    
    Returns
    -------
    pd.DataFrame
        Formatted test results table.
    """
    rows = []
    for model_name, metrics in test_results.items():
        rows.append({
            "Model": model_name,
            "MAE (kg/ha)": metrics.get("test_MAE", "N/A"),
            "MSE": metrics.get("test_MSE", "N/A"),
            "RMSE (kg/ha)": metrics.get("test_RMSE", "N/A"),
            "R²": metrics.get("test_R2", "N/A"),
        })
    return pd.DataFrame(rows)


def interpret_results(comparison, selected_model, selection_notes):
    """
    Generate plain-language interpretation of model comparison results.
    
    Unit 5: Results interpretation.
    
    Parameters
    ----------
    comparison : dict
        Model comparison results.
    selected_model : str
        Name of selected model.
    selection_notes : list
        Notes about model selection.
    
    Returns
    -------
    list
        List of interpretation strings.
    """
    interpretations = [
        f"**Selected Model: {selected_model}**",
        "",
        "Model selection was based on validation set MAE (lower is better).",
        "The test set was used ONCE for final reporting only.",
        "",
    ]
    
    for note in selection_notes:
        interpretations.append(note)
    
    return interpretations
