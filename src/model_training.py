"""
Model Training Module
=====================
Train Multiple Linear Regression, CART Regression Tree, and Random Forest.
Covers: Unit 3 (regression), Unit 4 (CART, pruning), Unit 5 (model combining).

Author: B.Tech AI & Data Science Student
Course: ISPA (21AIC401T)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import joblib
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import (
    load_and_validate_data, split_data, preprocess_data,
    FEATURE_DISPLAY, TARGET_DISPLAY, FEATURE_NAMES, RANDOM_STATE
)
from src.statistical_analysis import run_full_statistical_analysis

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, prefix=""):
    """
    Compute MAE, MSE, RMSE, R² evaluation metrics.
    
    Unit 5: Model evaluation metrics.
    
    Formulas:
    - MAE  = (1/n) Σ |yi - ŷi|
    - MSE  = (1/n) Σ (yi - ŷi)²
    - RMSE = √MSE
    - R²   = 1 - Σ(yi - ŷi)² / Σ(yi - ȳ)²
    
    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    prefix : str
        Prefix for metric keys.
    
    Returns
    -------
    dict
        Dictionary of metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        f"{prefix}MAE": round(float(mae), 2),
        f"{prefix}MSE": round(float(mse), 2),
        f"{prefix}RMSE": round(float(rmse), 2),
        f"{prefix}R2": round(float(r2), 4),
    }


def train_linear_regression(X_train, y_train, X_val, y_val):
    """
    Train Multiple Linear Regression (sklearn for prediction, statsmodels for inference).
    
    Unit 3: Statistical baseline model.
    
    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Processed (scaled) features.
    y_train, y_val : pd.Series
        Target values.
    
    Returns
    -------
    tuple
        (model, train_metrics, val_metrics)
    """
    print("\n--- Multiple Linear Regression (Unit 3) ---")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_metrics = compute_metrics(y_train, train_pred, "train_")
    val_metrics = compute_metrics(y_val, val_pred, "val_")
    
    # 5-fold cross-validation on training data (Unit 5)
    cv_scores = cross_val_score(
        LinearRegression(), X_train, y_train,
        cv=5, scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"  Train R²: {train_metrics['train_R2']:.4f}, Val R²: {val_metrics['val_R2']:.4f}")
    print(f"  Train MAE: {train_metrics['train_MAE']:.2f}, Val MAE: {val_metrics['val_MAE']:.2f}")
    print(f"  5-Fold CV MAE: {cv_mae:.2f} ± {cv_std:.2f}")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "linear_regression.joblib")
    joblib.dump(model, model_path)
    print(f"  Saved: {model_path}")
    
    return model, train_metrics, val_metrics, {"cv_mae": round(cv_mae, 2), "cv_std": round(cv_std, 2)}


def tune_cart_tree(X_train, y_train, X_val, y_val):
    """
    Train and tune CART Regression Tree with overfitting control.
    
    Unit 4: CART with pruning (max_depth, min_samples_leaf, ccp_alpha).
    
    CART splits nodes by minimizing Sum of Squared Errors (SSE):
    SSE = Σ (yi - ȳ_node)² for each child node
    
    Pruning controls complexity to prevent overfitting:
    - max_depth: limits tree depth
    - min_samples_leaf: minimum samples per leaf
    - ccp_alpha: cost-complexity pruning parameter
    
    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Processed features.
    y_train, y_val : pd.Series
        Target values.
    
    Returns
    -------
    tuple
        (best_model, train_metrics, val_metrics, tuning_results)
    """
    print("\n--- CART Regression Tree (Unit 4) ---")
    print("  Tuning hyperparameters on validation set...")
    
    # Hyperparameter grid (kept reasonable for a laptop)
    param_grid = []
    for depth in [3, 5, 7, 10, 15, None]:
        for min_leaf in [5, 10, 20, 50]:
            for alpha in [0.0, 0.001, 0.01, 0.1]:
                param_grid.append({
                    "max_depth": depth,
                    "min_samples_leaf": min_leaf,
                    "ccp_alpha": alpha,
                })
    
    best_val_mae = float("inf")
    best_params = None
    best_model = None
    tuning_log = []
    
    for params in param_grid:
        model = DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            **params
        )
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        tuning_log.append({
            "max_depth": params["max_depth"],
            "min_samples_leaf": params["min_samples_leaf"],
            "ccp_alpha": params["ccp_alpha"],
            "train_MAE": round(train_mae, 2),
            "val_MAE": round(val_mae, 2),
            "val_R2": round(val_r2, 4),
            "n_leaves": model.get_n_leaves(),
        })
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = params
            best_model = model
    
    # Retrain with best params (already done since best_model is saved)
    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)
    
    train_metrics = compute_metrics(y_train, train_pred, "train_")
    val_metrics = compute_metrics(y_val, val_pred, "val_")
    
    # 5-fold cross-validation on training data (Unit 5)
    cv_scores = cross_val_score(
        DecisionTreeRegressor(random_state=RANDOM_STATE, **best_params),
        X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"  Best params: depth={best_params['max_depth']}, "
          f"min_leaf={best_params['min_samples_leaf']}, "
          f"ccp_alpha={best_params['ccp_alpha']}")
    print(f"  Tree leaves: {best_model.get_n_leaves()}, depth: {best_model.get_depth()}")
    print(f"  Train R²: {train_metrics['train_R2']:.4f}, Val R²: {val_metrics['val_R2']:.4f}")
    print(f"  Train MAE: {train_metrics['train_MAE']:.2f}, Val MAE: {val_metrics['val_MAE']:.2f}")
    print(f"  5-Fold CV MAE: {cv_mae:.2f} ± {cv_std:.2f}")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "cart_tree.joblib")
    joblib.dump(best_model, model_path)
    print(f"  Saved: {model_path}")
    
    return best_model, train_metrics, val_metrics, {
        "best_params": best_params,
        "tuning_log": tuning_log,
        "n_leaves": best_model.get_n_leaves(),
        "tree_depth": best_model.get_depth(),
        "cv_mae": round(cv_mae, 2),
        "cv_std": round(cv_std, 2),
    }


def extract_rules(tree_model, feature_names, max_depth=5):
    """
    Extract IF-THEN rules from a trained CART tree.
    
    Unit 5: Rule induction from decision trees.
    
    Parameters
    ----------
    tree_model : DecisionTreeRegressor
        Trained tree.
    feature_names : list
        Feature names.
    max_depth : int
        Max depth for rule display.
    
    Returns
    -------
    str
        Text representation of tree rules.
    """
    display_names = [FEATURE_DISPLAY.get(f, f) for f in feature_names]
    rules = export_text(tree_model, feature_names=display_names, max_depth=max_depth)
    return rules


def plot_cart_tree(tree_model, feature_names, save_path=None):
    """
    Visualize the CART tree structure.
    
    Unit 4: Tree visualization.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "cart_tree.png")
    
    display_names = [FEATURE_DISPLAY.get(f, f) for f in feature_names]
    
    # Limit display depth for readability
    display_depth = min(tree_model.get_depth(), 4)
    
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        tree_model, feature_names=display_names,
        filled=True, rounded=True, ax=ax,
        max_depth=display_depth, fontsize=8,
        proportion=True, impurity=True,
    )
    ax.set_title(
        f"CART Regression Tree (showing depth ≤ {display_depth} of {tree_model.get_depth()})",
        fontsize=16
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


def plot_feature_importances(models_dict, feature_names, save_path=None):
    """
    Plot feature importances for tree-based models side by side.
    
    Unit 4/5: Feature importance analysis.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "feature_importances.png")
    
    display_names = [FEATURE_DISPLAY.get(f, f) for f in feature_names]
    n_models = len(models_dict)
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, model) in zip(axes, models_dict.items()):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
        ax.barh(
            [display_names[i] for i in indices],
            [importances[i] for i in indices],
            color=[colors[i] for i in range(len(indices))]
        )
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"{name} Feature Importances", fontsize=13)
        ax.invert_yaxis()
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest Regressor (EXTENSION: ensemble of CART trees).
    
    Unit 4 + Unit 5: An ensemble of many CART trees averaged.
    This is presented as an extension of CART (Unit 4) using the
    model combining concept (Unit 5).
    
    Random Forest = average prediction of B independent CART trees,
    each trained on a bootstrap sample with random feature subsets.
    
    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Processed features.
    y_train, y_val : pd.Series
        Target values.
    
    Returns
    -------
    tuple
        (model, train_metrics, val_metrics, tuning_info)
    """
    print("\n--- Random Forest (Extension: Unit 4 CART + Unit 5 Model Combining) ---")
    print("  NOTE: Random Forest is an ensemble of many CART trees averaged.")
    print("  Each tree is trained on a bootstrap sample with random feature subsets.")
    print("  Tuning on validation set...")
    
    # Modest grid (laptop-friendly)
    param_grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
        for n in [50, 100, 200]
        for d in [5, 10, 15, None]
        for l in [5, 10, 20]
    ]
    
    best_val_mae = float("inf")
    best_params = None
    best_model = None
    
    for params in param_grid:
        model = RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = params
            best_model = model
    
    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)
    
    train_metrics = compute_metrics(y_train, train_pred, "train_")
    val_metrics = compute_metrics(y_val, val_pred, "val_")
    
    # 5-fold cross-validation (Unit 5)
    cv_scores = cross_val_score(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best_params),
        X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"  Best params: n_trees={best_params['n_estimators']}, "
          f"depth={best_params['max_depth']}, min_leaf={best_params['min_samples_leaf']}")
    print(f"  Train R²: {train_metrics['train_R2']:.4f}, Val R²: {val_metrics['val_R2']:.4f}")
    print(f"  Train MAE: {train_metrics['train_MAE']:.2f}, Val MAE: {val_metrics['val_MAE']:.2f}")
    print(f"  5-Fold CV MAE: {cv_mae:.2f} ± {cv_std:.2f}")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "random_forest.joblib")
    joblib.dump(best_model, model_path)
    print(f"  Saved: {model_path}")
    
    return best_model, train_metrics, val_metrics, {
        "best_params": best_params,
        "cv_mae": round(cv_mae, 2),
        "cv_std": round(cv_std, 2),
    }


def run_full_training_pipeline():
    """
    Execute the complete training pipeline.
    
    This is the main entry point that:
    1. Loads and validates data
    2. Splits data (70/15/15)
    3. Runs statistical analysis
    4. Trains all models
    5. Generates all plots and outputs
    6. Saves everything to models/ and outputs/
    
    Returns
    -------
    dict
        Complete results dictionary.
    """
    print("="*60)
    print("CROP YIELD PREDICTION - FULL TRAINING PIPELINE")
    print("="*60)
    
    # 1. Load data
    print("\n[1/7] Loading and validating data...")
    df = load_and_validate_data()
    from src.data_preprocessing import data_quality_report, get_feature_ranges
    quality = data_quality_report(df)
    
    print(f"\nDescriptive Statistics:")
    print(quality["descriptive_stats"])
    
    # 2. Split data BEFORE preprocessing (Unit 4: leakage prevention)
    print("\n[2/7] Splitting data (70/15/15)...")
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(df)
    
    # Save raw training ranges for prediction warnings
    feature_ranges = get_feature_ranges(X_train_raw)
    joblib.dump(feature_ranges, os.path.join(MODEL_DIR, "feature_ranges.joblib"))
    
    # Keep full training data for ANOVA
    df_train_full = df.loc[X_train_raw.index].copy()
    
    # 3. Statistical analysis (on raw training data)
    print("\n[3/7] Running statistical analysis...")
    stat_results = run_full_statistical_analysis(X_train_raw, y_train, df_train_full)
    
    # 4. Preprocess (fit on train only)
    print("\n[4/7] Preprocessing data...")
    X_train, X_val, X_test, pipeline = preprocess_data(
        X_train_raw, X_val_raw, X_test_raw
    )
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "preprocessing_pipeline.joblib"))
    
    # Also save raw column order
    feature_cols = X_train.columns.tolist()
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.joblib"))
    
    # 5. Train models
    print("\n[5/7] Training models...")
    
    # 5a. Linear Regression (Unit 3)
    lr_model, lr_train, lr_val, lr_cv = train_linear_regression(
        X_train, y_train, X_val, y_val
    )
    
    # 5b. CART Regression Tree (Unit 4)
    cart_model, cart_train, cart_val, cart_info = tune_cart_tree(
        X_train, y_train, X_val, y_val
    )
    
    # 5c. Random Forest Extension (Unit 4+5)
    rf_model, rf_train, rf_val, rf_info = train_random_forest(
        X_train, y_train, X_val, y_val
    )
    
    # 6. Generate plots
    print("\n[6/7] Generating plots...")
    
    # CART tree plot
    tree_plot_path = plot_cart_tree(cart_model, feature_cols)
    
    # Feature importances
    fi_path = plot_feature_importances(
        {"CART": cart_model, "Random Forest": rf_model},
        feature_cols
    )
    
    # Extract CART rules (Unit 5: rule induction)
    rules_text = extract_rules(cart_model, feature_cols, max_depth=5)
    rules_path = os.path.join(OUTPUT_DIR, "cart_rules.txt")
    with open(rules_path, "w") as f:
        f.write(rules_text)
    print(f"  Saved: {rules_path}")
    
    # Actual vs Predicted plots
    plot_actual_vs_predicted(
        y_val, 
        {
            "Linear Regression": lr_model.predict(X_val),
            "CART Tree": cart_model.predict(X_val),
            "Random Forest": rf_model.predict(X_val),
        }
    )
    
    # Residual comparison plot
    plot_residual_comparison(
        y_val,
        {
            "Linear Regression": lr_model.predict(X_val),
            "CART Tree": cart_model.predict(X_val),
            "Random Forest": rf_model.predict(X_val),
        }
    )
    
    # Training vs Validation error plot (overfitting check)
    plot_overfitting_check(
        {
            "Linear Regression": (lr_train, lr_val),
            "CART Tree": (cart_train, cart_val),
            "Random Forest": (rf_train, rf_val),
        }
    )
    
    # 7. Compile results
    print("\n[7/7] Compiling results...")
    
    # Model comparison table
    comparison = {
        "Linear Regression": {**lr_train, **lr_val, **lr_cv, "unit": "3"},
        "CART Tree": {**cart_train, **cart_val, **cart_info, "unit": "4"},
        "Random Forest (Extension)": {**rf_train, **rf_val, **rf_info, "unit": "4+5"},
    }
    
    # Select best model based on validation MAE
    val_maes = {
        "Linear Regression": lr_val["val_MAE"],
        "CART Tree": cart_val["val_MAE"],
        "Random Forest (Extension)": rf_val["val_MAE"],
    }
    best_model_name = min(val_maes, key=val_maes.get)
    
    # Check if simpler model is nearly as good (within 5% of best)
    best_mae = val_maes[best_model_name]
    selection_notes = []
    for name, mae in sorted(val_maes.items(), key=lambda x: x[1]):
        pct_worse = ((mae - best_mae) / best_mae * 100) if best_mae > 0 else 0
        selection_notes.append(f"  {name}: MAE={mae:.2f} ({pct_worse:+.1f}% vs best)")
    
    # If CART is within 5% of RF, prefer CART for interpretability
    selected_model_name = best_model_name
    if best_model_name == "Random Forest (Extension)":
        cart_mae = val_maes["CART Tree"]
        if cart_mae <= best_mae * 1.05:  # Within 5%
            selected_model_name = "CART Tree"
            selection_notes.append(
                "  → Selected CART Tree over Random Forest: nearly as accurate "
                "but more interpretable (IF-THEN rules, visual tree)."
            )
        else:
            selection_notes.append(
                "  → Selected Random Forest: significantly more accurate than CART."
            )
    
    # Final test set evaluation (used ONCE!)
    print("\n--- FINAL TEST SET EVALUATION (used once) ---")
    models = {
        "Linear Regression": lr_model,
        "CART Tree": cart_model,
        "Random Forest (Extension)": rf_model,
    }
    test_results = {}
    for name, model in models.items():
        test_pred = model.predict(X_test)
        test_metrics = compute_metrics(y_test, test_pred, "test_")
        test_results[name] = test_metrics
        print(f"  {name}: MAE={test_metrics['test_MAE']:.2f}, "
              f"R²={test_metrics['test_R2']:.4f}")
    
    # Save OLS model for prediction intervals
    ols_model = stat_results.get("ols_model")
    if ols_model:
        joblib.dump(ols_model, os.path.join(MODEL_DIR, "ols_statsmodels.joblib"))
    
    all_results = {
        "data_quality": {
            "shape": quality["shape"],
            "missing": quality["missing_values"],
            "duplicates": quality["duplicates"],
            "outliers": quality["outliers"],
        },
        "descriptive_stats": quality["descriptive_stats"].to_dict(),
        "statistical_analysis": {
            "correlation": stat_results["correlation"],
            "anova_soil": stat_results["anova_soil"],
            "anova_fertilizer": stat_results["anova_fertilizer"],
            "regression_summary": stat_results["regression"]["summary"],
            "regression_interpretations": stat_results["regression"]["interpretations"],
        },
        "model_comparison": comparison,
        "test_results": test_results,
        "best_model_name": best_model_name,
        "selected_model_name": selected_model_name,
        "selection_notes": selection_notes,
        "cart_info": {
            "best_params": cart_info["best_params"],
            "n_leaves": cart_info["n_leaves"],
            "tree_depth": cart_info["tree_depth"],
            "rules_path": rules_path,
        },
        "rf_info": rf_info,
        "feature_ranges": feature_ranges,
        "feature_cols": feature_cols,
        "plots": {
            "heatmap": stat_results["correlation"]["heatmap_path"],
            "scatter": stat_results["correlation"]["scatter_path"],
            "residuals": stat_results["regression"]["residual_path"],
            "cart_tree": tree_plot_path,
            "feature_importances": fi_path,
            "anova_soil_plot": stat_results.get("anova_soil_plot"),
            "anova_fert_plot": stat_results.get("anova_fert_plot"),
        },
    }
    
    # Save complete results
    results_path = os.path.join(OUTPUT_DIR, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ All results saved to {results_path}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nSelected model: {selected_model_name}")
    print("Validation set rankings:")
    for note in selection_notes:
        print(note)
    
    return all_results


def plot_actual_vs_predicted(y_true, predictions_dict, save_path=None):
    """
    Actual vs Predicted scatter plots for all models.
    
    Unit 5: Model evaluation visualization.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    
    for ax, (name, y_pred) in zip(axes, predictions_dict.items()):
        ax.scatter(y_true, y_pred, alpha=0.2, s=10, color="steelblue")
        # Perfect prediction line
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())
        ]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ax.set_xlabel("Actual Yield (kg/ha)", fontsize=11)
        ax.set_ylabel("Predicted Yield (kg/ha)", fontsize=11)
        ax.set_title(f"{name}\nR²={r2:.4f}, MAE={mae:.0f}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    fig.suptitle("Actual vs Predicted (Validation Set)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


def plot_residual_comparison(y_true, predictions_dict, save_path=None):
    """
    Residual distribution plots for all models.
    
    Unit 5: Model comparison via residual analysis.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "residual_comparison.png")
    
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    
    for ax, (name, y_pred) in zip(axes, predictions_dict.items()):
        residuals = y_true.values - y_pred
        ax.hist(residuals, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Residual (Actual - Predicted) kg/ha", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"{name}\nMean residual: {np.mean(residuals):.0f}", fontsize=12)
        ax.grid(alpha=0.3)
    
    fig.suptitle("Residual Distributions (Validation Set)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


def plot_overfitting_check(metrics_dict, save_path=None):
    """
    Bar chart comparing training vs validation MAE for overfitting analysis.
    
    Unit 4/5: Overfitting detection.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "overfitting_check.png")
    
    models = list(metrics_dict.keys())
    train_maes = [metrics_dict[m][0]["train_MAE"] for m in models]
    val_maes = [metrics_dict[m][1]["val_MAE"] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, train_maes, width, label="Training MAE", color="#66B2FF")
    bars2 = ax.bar(x + width/2, val_maes, width, label="Validation MAE", color="#FF9999")
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("MAE (kg/ha)", fontsize=12)
    ax.set_title("Training vs Validation MAE (Overfitting Check)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{bar.get_height():.0f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{bar.get_height():.0f}", ha="center", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    results = run_full_training_pipeline()
