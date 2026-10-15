"""
Statistical Analysis Module
============================
Pearson correlation, ANOVA, OLS regression with full inference.
Covers: Unit 3 (correlation, regression, ANOVA, hypothesis testing).

Author: B.Tech AI & Data Science Student
Course: ISPA (21AIC401T)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, f_oneway, shapiro, levene
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import json
import warnings

from src.data_preprocessing import FEATURE_DISPLAY, TARGET_DISPLAY, FEATURE_NAMES

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pearson_correlation_analysis(X_train, y_train):
    """
    Compute Pearson correlation coefficient and p-value for each feature vs yield.
    
    Unit 3: Pearson correlation with hypothesis testing.
    
    Formula: r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² · Σ(yi - ȳ)²]
    H0: ρ = 0 (no linear correlation)
    H1: ρ ≠ 0 (significant linear correlation)
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (raw, not scaled).
    y_train : pd.Series
        Training target values.
    
    Returns
    -------
    pd.DataFrame
        Correlation results with columns: feature, r, p_value, significant, interpretation.
    """
    results = []
    for col in X_train.columns:
        r, p = pearsonr(X_train[col], y_train)
        # Interpret strength
        abs_r = abs(r)
        if abs_r < 0.1:
            strength = "negligible"
        elif abs_r < 0.3:
            strength = "weak"
        elif abs_r < 0.5:
            strength = "moderate"
        elif abs_r < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if r > 0 else "negative"
        
        display_name = FEATURE_DISPLAY.get(col, col)
        results.append({
            "feature": col,
            "display_name": display_name,
            "r": round(r, 4),
            "p_value": p,
            "significant": "Yes" if p < 0.05 else "No",
            "interpretation": f"{strength} {direction} correlation (r={r:.4f}, p={p:.2e})"
        })
    
    df_results = pd.DataFrame(results)
    print("✓ Pearson correlation analysis complete")
    print("  NOTE: Correlation does not imply causation. These values measure")
    print("  linear association only and do not establish cause-and-effect.")
    return df_results


def correlation_matrix(X_train, y_train):
    """
    Compute full correlation matrix including the target variable.
    
    Unit 3: Correlation analysis.
    
    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    df_combined = X_train.copy()
    df_combined["yield"] = y_train.values
    corr = df_combined.corr()
    return corr


def plot_correlation_heatmap(corr_matrix, save_path=None):
    """
    Plot a correlation heatmap with annotations.
    
    Unit 3: Visual representation of correlations.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    
    # Rename for display
    rename = {k: FEATURE_DISPLAY.get(k, k) for k in corr_matrix.columns}
    rename["yield"] = TARGET_DISPLAY
    corr_display = corr_matrix.rename(index=rename, columns=rename)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_display, dtype=bool), k=1)
    sns.heatmap(
        corr_display, annot=True, fmt=".3f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, mask=mask,
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Pearson Correlation Coefficient"}
    )
    ax.set_title("Pearson Correlation Matrix (Training Data)", fontsize=14, pad=15)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


def plot_scatter_plots(X_train, y_train, save_path=None):
    """
    Scatter plot of each feature vs yield with regression line.
    
    Unit 3: Visual correlation analysis.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "scatter_plots.png")
    
    features = X_train.columns.tolist()
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    
    for i, col in enumerate(features):
        ax = axes[i]
        ax.scatter(X_train[col], y_train, alpha=0.2, s=10, color="steelblue")
        # Add regression line
        z = np.polyfit(X_train[col], y_train, 1)
        p = np.poly1d(z)
        x_line = np.linspace(X_train[col].min(), X_train[col].max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2, label="Trend line")
        
        r, pval = pearsonr(X_train[col], y_train)
        ax.set_xlabel(FEATURE_DISPLAY.get(col, col), fontsize=11)
        ax.set_ylabel(TARGET_DISPLAY if i == 0 else "", fontsize=11)
        ax.set_title(f"r = {r:.3f}, p = {pval:.2e}", fontsize=10)
        ax.legend(fontsize=8)
    
    fig.suptitle("Feature vs Yield Scatter Plots (Training Data)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


def anova_analysis(df_train_raw, y_train, group_col="soil_type"):
    """
    One-way ANOVA: compare mean yield across groups (e.g., soil types).
    
    Unit 3: ANOVA hypothesis testing.
    
    H0: μ1 = μ2 = ... = μk (all group means are equal)
    H1: At least one group mean is different
    
    F = (Between-group variance) / (Within-group variance)
    
    Checks assumptions:
    1. Normality within each group (Shapiro-Wilk on sample)
    2. Homogeneity of variances (Levene's test)
    
    Parameters
    ----------
    df_train_raw : pd.DataFrame
        Raw training data including the grouping column.
    y_train : pd.Series
        Training target values.
    group_col : str
        Column name for grouping.
    
    Returns
    -------
    dict
        ANOVA results including F-statistic, p-value, group stats, assumptions.
    """
    if group_col not in df_train_raw.columns:
        return {"applicable": False, "reason": f"Column '{group_col}' not found in data."}
    
    # Combine for grouping
    df = df_train_raw.copy()
    df["yield"] = y_train.values
    
    groups = df[group_col].unique()
    if len(groups) < 2:
        return {"applicable": False, "reason": f"Need ≥2 groups, found {len(groups)}."}
    
    group_data = [df[df[group_col] == g]["yield"].values for g in groups]
    
    # Group statistics
    group_stats = []
    for g, data in zip(groups, group_data):
        group_stats.append({
            "group": str(g),
            "n": len(data),
            "mean": round(float(np.mean(data)), 2),
            "std": round(float(np.std(data, ddof=1)), 2),
            "median": round(float(np.median(data)), 2),
        })
    
    # Assumption 1: Normality (Shapiro-Wilk on sample of max 500)
    normality_results = []
    for g, data in zip(groups, group_data):
        sample = data[:500] if len(data) > 500 else data
        stat, p = shapiro(sample)
        normality_results.append({
            "group": str(g), "W": round(stat, 4), "p": round(p, 6),
            "normal": "Yes" if p > 0.05 else "No"
        })
    
    # Assumption 2: Homogeneity of variances (Levene's test)
    lev_stat, lev_p = levene(*group_data)
    
    # Perform ANOVA
    f_stat, p_value = f_oneway(*group_data)
    
    result = {
        "applicable": True,
        "group_col": group_col,
        "n_groups": len(groups),
        "groups": [str(g) for g in groups],
        "group_stats": group_stats,
        "normality": normality_results,
        "levene_stat": round(float(lev_stat), 4),
        "levene_p": round(float(lev_p), 6),
        "equal_variances": "Yes" if lev_p > 0.05 else "No",
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(p_value),
        "significant": "Yes" if p_value < 0.05 else "No",
        "interpretation": (
            f"F({len(groups)-1}, {sum(len(d) for d in group_data)-len(groups)}) = {f_stat:.4f}, "
            f"p = {p_value:.2e}. "
            + ("Reject H0: At least one group mean is significantly different."
               if p_value < 0.05 else
               "Fail to reject H0: No significant difference between group means.")
        ),
    }
    
    print(f"✓ ANOVA ({group_col}): F={f_stat:.4f}, p={p_value:.2e}")
    return result


def plot_anova_boxplot(df_train_raw, y_train, group_col="soil_type", save_path=None):
    """
    Box plot of yield by group for ANOVA visualization.
    
    Unit 3: ANOVA visual representation.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, f"anova_boxplot_{group_col}.png")
    
    if group_col not in df_train_raw.columns:
        return None
    
    df = df_train_raw.copy()
    df["yield"] = y_train.values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = sorted(df[group_col].unique())
    data_by_group = [df[df[group_col] == g]["yield"].values for g in groups]
    
    bp = ax.boxplot(data_by_group, labels=groups, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99", "#FF99CC", "#99CCFF"]
    for patch, color in zip(bp["boxes"], colors[:len(groups)]):
        patch.set_facecolor(color)
    
    ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(TARGET_DISPLAY, fontsize=12)
    ax.set_title(f"Yield Distribution by {group_col.replace('_', ' ').title()} (ANOVA)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


def anova_fertilizer_levels(X_train, y_train):
    """
    ANOVA on fertilizer levels (Low/Medium/High by tertiles).
    
    Unit 3: ANOVA on derived groups.
    """
    if "fertilizer" not in X_train.columns:
        return {"applicable": False, "reason": "Fertilizer column not found."}
    
    df = X_train.copy()
    df["yield"] = y_train.values
    
    # Create tertile groups
    df["fertilizer_level"] = pd.qcut(
        df["fertilizer"], q=3, labels=["Low", "Medium", "High"]
    )
    
    groups = ["Low", "Medium", "High"]
    group_data = [df[df["fertilizer_level"] == g]["yield"].values for g in groups]
    
    f_stat, p_value = f_oneway(*group_data)
    
    group_stats = []
    for g, data in zip(groups, group_data):
        group_stats.append({
            "group": g,
            "n": len(data),
            "mean": round(float(np.mean(data)), 2),
            "std": round(float(np.std(data, ddof=1)), 2),
        })
    
    result = {
        "applicable": True,
        "group_col": "Fertilizer Level (Tertiles)",
        "n_groups": 3,
        "groups": groups,
        "group_stats": group_stats,
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(p_value),
        "significant": "Yes" if p_value < 0.05 else "No",
        "interpretation": (
            f"F(2, {sum(len(d) for d in group_data)-3}) = {f_stat:.4f}, "
            f"p = {p_value:.2e}. "
            + ("Reject H0: Mean yield differs significantly across fertilizer levels."
               if p_value < 0.05 else
               "Fail to reject H0: No significant yield difference across fertilizer levels.")
        ),
    }
    
    print(f"✓ ANOVA (Fertilizer Levels): F={f_stat:.4f}, p={p_value:.2e}")
    return result


def ols_regression(X_train, y_train):
    """
    Fit Ordinary Least Squares (OLS) multiple linear regression.
    
    Unit 3: Regression inference.
    
    Model: y = β0 + β1·x1 + β2·x2 + ... + βk·xk + ε
    
    Reports:
    - Fitted equation
    - Coefficients, standard errors, t-statistics, p-values
    - 95% confidence intervals for each coefficient
    - R², adjusted R²
    - F-test for overall model significance
    - VIF (Variance Inflation Factor) for multicollinearity
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (raw, not scaled, for interpretable coefficients).
    y_train : pd.Series
        Training target.
    
    Returns
    -------
    tuple
        (statsmodels OLS results object, summary dict)
    """
    # Add constant (intercept) for statsmodels
    X_with_const = sm.add_constant(X_train, has_constant="add")
    
    model = sm.OLS(y_train, X_with_const).fit()
    
    # Build summary dictionary
    coef_summary = []
    params = model.params
    conf_int = model.conf_int(alpha=0.05)
    
    for i, name in enumerate(X_with_const.columns):
        display = FEATURE_DISPLAY.get(name, name)
        if name == "const":
            display = "Intercept"
        coef_summary.append({
            "variable": name,
            "display_name": display,
            "coefficient": round(float(params.iloc[i]), 4),
            "std_error": round(float(model.bse.iloc[i]), 4),
            "t_statistic": round(float(model.tvalues.iloc[i]), 4),
            "p_value": float(model.pvalues.iloc[i]),
            "ci_lower": round(float(conf_int.iloc[i, 0]), 4),
            "ci_upper": round(float(conf_int.iloc[i, 1]), 4),
            "significant": "Yes" if model.pvalues.iloc[i] < 0.05 else "No",
        })
    
    # VIF (only for features, not intercept)
    vif_data = []
    feature_cols = [c for c in X_with_const.columns if c != "const"]
    X_features_only = X_with_const[feature_cols]
    for i, col in enumerate(feature_cols):
        vif_val = variance_inflation_factor(X_features_only.values, i)
        vif_data.append({
            "feature": col,
            "display_name": FEATURE_DISPLAY.get(col, col),
            "vif": round(float(vif_val), 4),
            "multicollinearity": "High" if vif_val > 10 else ("Moderate" if vif_val > 5 else "Low"),
        })
    
    # Build the equation string
    eq_parts = [f"{params.iloc[0]:.2f}"]
    for i, name in enumerate(feature_cols):
        coef = params.iloc[i + 1]
        sign = " + " if coef >= 0 else " - "
        eq_parts.append(f"{sign}{abs(coef):.2f}·{FEATURE_DISPLAY.get(name, name)}")
    equation = f"Yield = {''.join(eq_parts)}"
    
    summary = {
        "equation": equation,
        "coefficients": coef_summary,
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "f_statistic": round(float(model.fvalue), 4),
        "f_pvalue": float(model.f_pvalue),
        "aic": round(float(model.aic), 2),
        "bic": round(float(model.bic), 2),
        "n_obs": int(model.nobs),
        "df_model": int(model.df_model),
        "df_resid": int(model.df_resid),
        "vif": vif_data,
        "ols_summary_text": str(model.summary()),
    }
    
    print(f"✓ OLS Regression: R²={model.rsquared:.4f}, Adj.R²={model.rsquared_adj:.4f}")
    print(f"  F-test: F={model.fvalue:.2f}, p={model.f_pvalue:.2e}")
    print(f"  Equation: {equation}")
    
    return model, summary


def prediction_interval(ols_model, X_new, alpha=0.05):
    """
    Compute prediction interval for new observations.
    
    Unit 3: Prediction intervals.
    
    The prediction interval is wider than a confidence interval because it
    accounts for both the uncertainty in the estimated mean and the variability
    of individual observations.
    
    Parameters
    ----------
    ols_model : statsmodels OLS results
        Fitted OLS model.
    X_new : pd.DataFrame
        New feature values (raw).
    alpha : float
        Significance level (default 0.05 for 95% intervals).
    
    Returns
    -------
    pd.DataFrame
        Predictions with confidence and prediction intervals.
    """
    X_new_const = sm.add_constant(X_new, has_constant="add")
    predictions = ols_model.get_prediction(X_new_const)
    pred_summary = predictions.summary_frame(alpha=alpha)
    
    result = pd.DataFrame({
        "predicted": pred_summary["mean"].values,
        "ci_lower": pred_summary["mean_ci_lower"].values,
        "ci_upper": pred_summary["mean_ci_upper"].values,
        "pi_lower": pred_summary["obs_ci_lower"].values,
        "pi_upper": pred_summary["obs_ci_upper"].values,
    })
    
    return result


def plot_residuals(ols_model, X_train, y_train, save_path=None):
    """
    Plot residual diagnostics: residuals vs fitted and Q-Q plot.
    
    Unit 3: Regression diagnostics.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "residual_plots.png")
    
    X_with_const = sm.add_constant(X_train, has_constant="add")
    y_pred = ols_model.predict(X_with_const)
    residuals = y_train.values - y_pred.values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Fitted
    axes[0].scatter(y_pred, residuals, alpha=0.2, s=10, color="steelblue")
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Fitted Values (kg/ha)", fontsize=11)
    axes[0].set_ylabel("Residuals (kg/ha)", fontsize=11)
    axes[0].set_title("Residuals vs Fitted Values", fontsize=13)
    axes[0].grid(alpha=0.3)
    
    # Q-Q Plot
    sm.qqplot(residuals, line="45", ax=axes[1], alpha=0.3, markersize=3)
    axes[1].set_title("Normal Q-Q Plot of Residuals", fontsize=13)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


def interpret_coefficients(summary_dict):
    """
    Generate plain-language interpretation of regression coefficients.
    
    Unit 3: Statistical interpretation.
    
    Returns
    -------
    list
        List of interpretation strings.
    """
    interpretations = []
    for coef in summary_dict["coefficients"]:
        if coef["variable"] == "const":
            interpretations.append(
                f"**Intercept** = {coef['coefficient']:.2f}: The predicted yield when all "
                f"features are zero (theoretical baseline, may not be practically meaningful)."
            )
        else:
            direction = "increases" if coef["coefficient"] > 0 else "decreases"
            sig_note = ("statistically significant (p < 0.05)"
                       if coef["significant"] == "Yes"
                       else "NOT statistically significant (p ≥ 0.05)")
            interpretations.append(
                f"**{coef['display_name']}** (β = {coef['coefficient']:.2f}): "
                f"For each one-unit increase in {coef['display_name'].lower()}, "
                f"yield {direction} by {abs(coef['coefficient']):.2f} kg/ha on average, "
                f"holding other variables constant. 95% CI: [{coef['ci_lower']:.2f}, {coef['ci_upper']:.2f}]. "
                f"This coefficient is {sig_note}."
            )
    return interpretations


def run_full_statistical_analysis(X_train, y_train, df_train_raw=None):
    """
    Run the complete statistical analysis pipeline.
    
    Unit 3: Complete statistical analysis.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Raw training features.
    y_train : pd.Series
        Training target.
    df_train_raw : pd.DataFrame, optional
        Full training data including categorical columns for ANOVA.
    
    Returns
    -------
    dict
        All analysis results.
    """
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS (Unit 3)")
    print("="*60)
    
    # 1. Pearson Correlation
    print("\n--- Pearson Correlation ---")
    corr_results = pearson_correlation_analysis(X_train, y_train)
    corr_mat = correlation_matrix(X_train, y_train)
    heatmap_path = plot_correlation_heatmap(corr_mat)
    scatter_path = plot_scatter_plots(X_train, y_train)
    
    # 2. ANOVA
    print("\n--- ANOVA ---")
    anova_soil = {"applicable": False, "reason": "No raw data with soil_type provided."}
    anova_soil_plot = None
    anova_fert = {"applicable": False, "reason": "No fertilizer data."}
    
    if df_train_raw is not None and "soil_type" in df_train_raw.columns:
        anova_soil = anova_analysis(df_train_raw, y_train, "soil_type")
        anova_soil_plot = plot_anova_boxplot(df_train_raw, y_train, "soil_type")
    
    anova_fert = anova_fertilizer_levels(X_train, y_train)
    
    # ANOVA boxplot for fertilizer levels
    anova_fert_plot = None
    if "fertilizer" in X_train.columns:
        df_temp = X_train.copy()
        df_temp["yield"] = y_train.values
        df_temp["fertilizer_level"] = pd.qcut(df_temp["fertilizer"], q=3, labels=["Low", "Medium", "High"])
        anova_fert_plot = plot_anova_boxplot(
            df_temp, y_train, "fertilizer_level",
            save_path=os.path.join(OUTPUT_DIR, "anova_boxplot_fertilizer_level.png")
        )
    
    # 3. OLS Regression
    print("\n--- OLS Multiple Linear Regression ---")
    ols_model, ols_summary = ols_regression(X_train, y_train)
    residual_path = plot_residuals(ols_model, X_train, y_train)
    interpretations = interpret_coefficients(ols_summary)
    
    results = {
        "correlation": {
            "results": corr_results.to_dict("records"),
            "matrix": corr_mat.to_dict(),
            "heatmap_path": heatmap_path,
            "scatter_path": scatter_path,
        },
        "anova_soil": anova_soil,
        "anova_soil_plot": anova_soil_plot,
        "anova_fertilizer": anova_fert,
        "anova_fert_plot": anova_fert_plot,
        "regression": {
            "summary": ols_summary,
            "residual_path": residual_path,
            "interpretations": interpretations,
        },
        "ols_model": ols_model,
    }
    
    # Save results (excluding non-serializable objects)
    save_results = {k: v for k, v in results.items() if k != "ols_model"}
    # Convert paths to strings
    results_path = os.path.join(OUTPUT_DIR, "statistical_analysis.json")
    try:
        with open(results_path, "w") as f:
            json.dump(save_results, f, indent=2, default=str)
        print(f"\n✓ Results saved to {results_path}")
    except Exception as e:
        print(f"  Warning: Could not save JSON: {e}")
    
    return results


if __name__ == "__main__":
    from src.data_preprocessing import load_and_validate_data, split_data
    
    df = load_and_validate_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # Keep raw training data for ANOVA (with categorical columns)
    df_train_raw = df.loc[X_train.index].copy()
    
    results = run_full_statistical_analysis(X_train, y_train, df_train_raw)
