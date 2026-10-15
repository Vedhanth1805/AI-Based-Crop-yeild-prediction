"""
Streamlit App - AI-Based Crop Yield Prediction
===============================================
5-page app with sidebar navigation:
1. Dashboard
2. Statistical Analysis
3. Model Training & Comparison
4. Yield Prediction
5. Insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Page Config ----
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Constants ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

from src.data_preprocessing import (
    FEATURE_DISPLAY, TARGET_DISPLAY, FEATURE_NAMES,
    DATA_SOURCE, COLUMN_MAP, load_and_validate_data, data_quality_report
)


# ---- Helper: Load Results ----
@st.cache_data
def load_training_results():
    """Load the training results JSON."""
    path = os.path.join(OUTPUT_DIR, "training_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_dataset():
    """Load the dataset."""
    try:
        return load_and_validate_data()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def load_image(filename):
    """Load an image from outputs directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return path
    return None


# ---- Sidebar Navigation ----
st.sidebar.title("🌾 Crop Yield Prediction")
st.sidebar.markdown("---")

pages = [
    "🔮 Predict Yield",
    "📊 Dashboard",
    "📈 Statistical Analysis",
    "🤖 Model Training & Comparison",
    "💡 Insights",
]
page = st.sidebar.radio("Navigate to:", pages)

results = load_training_results()

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================
if page == "📊 Dashboard":
    st.title("🌾 AI-Based Crop Yield Prediction")
    st.markdown("### Using Machine Learning and Statistical Analysis")
    
    # Data source banner
    if DATA_SOURCE == "REAL":
        st.success(
            "✅ **Real Dataset** — This project uses a real agricultural dataset "
            "(10,000 crop records with rainfall, temperature, soil pH, fertilizer, and yield)."
        )
    else:
        st.warning(
            "⚠️ **SIMULATED DATA FOR DEMONSTRATION** — This project uses synthetic data. "
            "Replace data/crop_data.csv with a real dataset and re-run the pipeline."
        )
    
    st.markdown("""
    **Objective:** Predict crop yield (kg/hectare) from environmental and agricultural factors 
    (rainfall, temperature, soil pH, fertilizer amount). Analyse how these factors relate to 
    yield using statistical and machine learning methods.
    
    **Algorithms Used:**
    - Multiple Linear Regression
    - CART Regression Tree
    - Random Forest Regressor
    """)
    
    # Dataset Overview
    st.markdown("---")
    st.subheader("📋 Dataset Overview")
    
    df = load_dataset()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Features Used", len([f for f in FEATURE_NAMES if f in df.columns]))
        col3.metric("Target", "Yield (kg/ha)")
        col4.metric("Missing Values", df[FEATURE_NAMES + ["yield"]].isnull().sum().sum())
        
        # Show first few rows
        st.markdown("**Sample Data (first 5 rows):**")
        display_cols = [f for f in FEATURE_NAMES if f in df.columns] + ["yield"]
        display_df = df[display_cols].head()
        display_df.columns = [FEATURE_DISPLAY.get(c, c) if c != "yield" else TARGET_DISPLAY 
                              for c in display_cols]
        st.dataframe(display_df, use_container_width=True)
        
        # Descriptive statistics
        st.markdown("**Descriptive Statistics:**")
        desc = df[display_cols].describe().round(2)
        desc.columns = [FEATURE_DISPLAY.get(c, c) if c != "yield" else TARGET_DISPLAY 
                        for c in display_cols]
        st.dataframe(desc, use_container_width=True)
        
        # Data quality
        if results:
            st.markdown("**Data Quality:**")
            quality = results.get("data_quality", {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Outliers (IQR Method):**")
                outlier_data = []
                for col, info in quality.get("outliers", {}).items():
                    outlier_data.append({
                        "Feature": FEATURE_DISPLAY.get(col, col),
                        "Outliers": info.get("n_outliers", 0),
                        "% of Data": info.get("pct_outliers", 0),
                        "IQR Range": f"[{info.get('lower_bound', '')}, {info.get('upper_bound', '')}]",
                    })
                if outlier_data:
                    st.dataframe(pd.DataFrame(outlier_data), use_container_width=True)
            
            with col2:
                st.markdown("**Missing Values:**")
                missing = quality.get("missing", {})
                missing_df = pd.DataFrame({
                    "Feature": [FEATURE_DISPLAY.get(k, k) for k in missing.keys()],
                    "Missing Count": list(missing.values()),
                })
                st.dataframe(missing_df, use_container_width=True)
    
    if results:
        st.markdown("---")
        st.subheader("🏆 Key Results Summary")
        
        selected = results.get("selected_model_name", "N/A")
        test_res = results.get("test_results", {})
        
        col1, col2, col3 = st.columns(3)
        
        if selected in test_res:
            metrics = test_res[selected]
            col1.metric("Selected Model", selected)
            col2.metric("Test MAE", f"{metrics.get('test_MAE', 'N/A')} kg/ha")
            col3.metric("Test R²", f"{metrics.get('test_R2', 'N/A')}")


# ============================================================================
# PAGE 2: STATISTICAL ANALYSIS
# ============================================================================
elif page == "📈 Statistical Analysis":
    st.title("📈 Statistical Analysis")
    
    if results is None:
        st.error("Training results not found. Run `python -m src.model_training` first.")
        st.stop()
    
    stat = results.get("statistical_analysis", {})
    
    # Pearson Correlation
    st.subheader("1. Pearson Correlation Analysis")
    st.markdown("""
    **Pearson correlation coefficient (r)** measures the linear relationship between two variables.
    - r = +1: perfect positive linear relationship
    - r = 0: no linear relationship  
    - r = -1: perfect negative linear relationship
    
    **H₀:** ρ = 0 (no linear correlation) | **H₁:** ρ ≠ 0 (significant linear correlation)
    """)
    
    corr = stat.get("correlation", {})
    if corr.get("results"):
        corr_df = pd.DataFrame(corr["results"])
        display_df = corr_df[["display_name", "r", "p_value", "significant", "interpretation"]].copy()
        display_df.columns = ["Feature", "r", "p-value", "Significant (α=0.05)", "Interpretation"]
        display_df["p-value"] = display_df["p-value"].apply(lambda x: f"{x:.2e}")
        st.dataframe(display_df, use_container_width=True)
    
    st.info(
        "**Note:** Correlation does NOT imply causation. These values measure linear association "
        "only. A high correlation between rainfall and yield does not mean rainfall *causes* "
        "higher yield — there may be confounding variables or reverse causality."
    )
    
    # Heatmap and Scatter plots
    col1, col2 = st.columns(2)
    with col1:
        heatmap = load_image("correlation_heatmap.png")
        if heatmap:
            st.image(heatmap, caption="Correlation Heatmap", use_container_width=True)
    with col2:
        scatter = load_image("scatter_plots.png")
        if scatter:
            st.image(scatter, caption="Feature vs Yield Scatter Plots", use_container_width=True)
    
    # ANOVA
    st.markdown("---")
    st.subheader("2. One-Way ANOVA")
    st.markdown("""
    **ANOVA** tests whether the mean yield differs significantly across groups.
    - **H₀:** μ₁ = μ₂ = ... = μₖ (all group means equal)
    - **H₁:** At least one group mean is different
    - **F-statistic** = Between-group variance / Within-group variance
    """)
    
    # ANOVA by Soil Type
    anova_soil = stat.get("anova_soil", {})
    if anova_soil.get("applicable"):
        st.markdown("#### ANOVA: Yield by Soil Type")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            soil_plot = load_image("anova_boxplot_soil_type.png")
            if soil_plot:
                st.image(soil_plot, use_container_width=True)
        with col2:
            st.metric("F-statistic", f"{anova_soil.get('f_statistic', 'N/A')}")
            st.metric("p-value", f"{anova_soil.get('p_value', 0):.2e}")
            st.metric("Significant?", anova_soil.get("significant", "N/A"))
            
            if anova_soil.get("group_stats"):
                st.markdown("**Group Means:**")
                gs_df = pd.DataFrame(anova_soil["group_stats"])
                st.dataframe(gs_df, use_container_width=True)
        
        st.markdown(f"**Interpretation:** {anova_soil.get('interpretation', 'N/A')}")
        
        # Assumptions
        with st.expander("ANOVA Assumptions Check"):
            st.markdown(f"**Levene's Test (Equal Variances):** "
                       f"W = {anova_soil.get('levene_stat', 'N/A')}, "
                       f"p = {anova_soil.get('levene_p', 'N/A')} — "
                       f"Equal variances: {anova_soil.get('equal_variances', 'N/A')}")
            if anova_soil.get("normality"):
                st.markdown("**Shapiro-Wilk Normality Test (per group):**")
                norm_df = pd.DataFrame(anova_soil["normality"])
                st.dataframe(norm_df, use_container_width=True)
    else:
        st.info(f"Soil Type ANOVA: {anova_soil.get('reason', 'Not applicable')}")
    
    # ANOVA by Fertilizer Level
    anova_fert = stat.get("anova_fertilizer", {})
    if anova_fert.get("applicable"):
        st.markdown("#### ANOVA: Yield by Fertilizer Level (Tertiles)")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fert_plot = load_image("anova_boxplot_fertilizer_level.png")
            if fert_plot:
                st.image(fert_plot, use_container_width=True)
        with col2:
            st.metric("F-statistic", f"{anova_fert.get('f_statistic', 'N/A')}")
            st.metric("p-value", f"{anova_fert.get('p_value', 0):.2e}")
            st.metric("Significant?", anova_fert.get("significant", "N/A"))
        
        st.markdown(f"**Interpretation:** {anova_fert.get('interpretation', 'N/A')}")
    
    # OLS Regression
    st.markdown("---")
    st.subheader("3. Multiple Linear Regression (OLS)")
    
    reg = stat.get("regression_summary", {})
    if reg:
        st.markdown(f"**Fitted Equation:** `{reg.get('equation', 'N/A')}`")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R²", f"{reg.get('r_squared', 'N/A')}")
        col2.metric("Adjusted R²", f"{reg.get('adj_r_squared', 'N/A')}")
        col3.metric("F-statistic", f"{reg.get('f_statistic', 'N/A')}")
        col4.metric("F p-value", f"{reg.get('f_pvalue', 0):.2e}")
        
        # Coefficients table
        st.markdown("**Coefficients:**")
        if reg.get("coefficients"):
            coef_df = pd.DataFrame(reg["coefficients"])
            display_coef = coef_df[["display_name", "coefficient", "std_error",
                                     "t_statistic", "p_value", "ci_lower", "ci_upper", "significant"]].copy()
            display_coef.columns = ["Variable", "β", "Std Error", "t", "p-value", 
                                    "95% CI Lower", "95% CI Upper", "Significant"]
            display_coef["p-value"] = display_coef["p-value"].apply(lambda x: f"{x:.2e}")
            st.dataframe(display_coef, use_container_width=True)
        
        # VIF
        if reg.get("vif"):
            st.markdown("**Variance Inflation Factor (VIF) — Multicollinearity Check:**")
            vif_df = pd.DataFrame(reg["vif"])
            vif_display = vif_df[["display_name", "vif", "multicollinearity"]].copy()
            vif_display.columns = ["Feature", "VIF", "Level"]
            st.dataframe(vif_display, use_container_width=True)
            st.markdown("VIF > 10: high multicollinearity concern. VIF > 5: moderate. VIF < 5: acceptable.")
        
        # Plain-language interpretation
        st.markdown("**Plain-Language Interpretation:**")
        interpretations = stat.get("regression_interpretations", [])
        for interp in interpretations:
            st.markdown(f"- {interp}")
        
        # Residual plots
        residual_img = load_image("residual_plots.png")
        if residual_img:
            st.image(residual_img, caption="Residual Diagnostics", use_container_width=True)
        
        # Full OLS summary (expandable)
        with st.expander("Full OLS Summary (statsmodels)"):
            st.code(reg.get("ols_summary_text", "Not available"), language="text")


# ============================================================================
# PAGE 3: MODEL TRAINING & COMPARISON
# ============================================================================
elif page == "🤖 Model Training & Comparison":
    st.title("🤖 Model Training & Comparison")
    
    if results is None:
        st.error("Training results not found. Run `python -m src.model_training` first.")
        st.stop()
    
    # Split methodology
    st.subheader("1. Data Splitting Methodology")
    st.markdown("""
    **70% Training / 15% Validation / 15% Test** split with `random_state=42`.
    
    ⚠️ **Data Leakage Prevention:**
    - Split is done BEFORE any imputation or scaling.
    - The preprocessing pipeline is fitted on training data ONLY.
    - Validation set: used for hyperparameter tuning and model selection.
    - Test set: used ONCE for final reporting (never during model development).
    """)
    
    # Model descriptions
    st.subheader("2. Models Trained")
    
    tab1, tab2, tab3 = st.tabs([
        "Linear Regression", 
        "CART Regression Tree", 
        "Random Forest (Extension)"
    ])
    
    with tab1:
        st.markdown("""
        **Multiple Linear Regression** is a statistical baseline model.
        - Model: $y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_k x_k + \\epsilon$
        - Assumes linear relationship between features and target.
        - Coefficients have direct statistical interpretation.
        - Provides confidence and prediction intervals.
        """)
    
    with tab2:
        st.markdown("""
        **CART (Classification and Regression Trees)** is the main ML model.
        - Splits data by minimising Sum of Squared Errors (SSE) at each node.
        - **Pruning** controls overfitting: `max_depth`, `min_samples_leaf`, `ccp_alpha`.
        - Produces interpretable IF-THEN rules.
        - Non-parametric: captures non-linear patterns.
        """)
        
        cart_info = results.get("cart_info", {})
        if cart_info:
            best_p = cart_info.get("best_params", {})
            st.markdown(f"""
            **Best Hyperparameters (tuned on validation set):**
            - `max_depth` = {best_p.get('max_depth', 'None')}
            - `min_samples_leaf` = {best_p.get('min_samples_leaf', 'N/A')}
            - `ccp_alpha` = {best_p.get('ccp_alpha', 'N/A')}
            - Tree depth = {cart_info.get('tree_depth', 'N/A')}, Leaves = {cart_info.get('n_leaves', 'N/A')}
            """)
        
        # Tree plot
        tree_img = load_image("cart_tree.png")
        if tree_img:
            st.image(tree_img, caption="CART Regression Tree (partial view)", use_container_width=True)
        
        # Rules
        rules_path = os.path.join(OUTPUT_DIR, "cart_rules.txt")
        if os.path.exists(rules_path):
            with st.expander("IF-THEN Rules"):
                with open(rules_path) as f:
                    rules = f.read()
                st.code(rules, language="text")
    
    with tab3:
        st.markdown("""
        **Random Forest** is an ensemble extension of CART.
        
        It is an **ensemble of many CART trees averaged**:
        - Each tree is trained on a bootstrap sample (random subset with replacement).
        - At each split, only a random subset of features is considered.
        - Final prediction = average of all individual tree predictions.
        - Reduces variance compared to a single CART tree.
        
        Combining multiple trees helps reduce variance compared to a single CART tree.
        """)
        
        rf_info = results.get("rf_info", {})
        if rf_info:
            best_p = rf_info.get("best_params", {})
            st.markdown(f"""
            **Best Hyperparameters:**
            - `n_estimators` = {best_p.get('n_estimators', 'N/A')} trees
            - `max_depth` = {best_p.get('max_depth', 'None')}
            - `min_samples_leaf` = {best_p.get('min_samples_leaf', 'N/A')}
            """)
    
    # Metrics comparison
    st.markdown("---")
    st.subheader("3. Model Comparison")
    
    comparison = results.get("model_comparison", {})
    if comparison:
        # Build comparison table
        rows = []
        for model_name, metrics in comparison.items():
            rows.append({
                "Model": model_name,
                "Train MAE": metrics.get("train_MAE", ""),
                "Train RMSE": metrics.get("train_RMSE", ""),
                "Train R²": metrics.get("train_R2", ""),
                "Val MAE": metrics.get("val_MAE", ""),
                "Val RMSE": metrics.get("val_RMSE", ""),
                "Val R²": metrics.get("val_R2", ""),
                "CV MAE (5-fold)": metrics.get("cv_mae", ""),
            })
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df, use_container_width=True)
    
    # Final test results
    st.markdown("**Final Test Set Results (used ONCE):**")
    test_res = results.get("test_results", {})
    if test_res:
        test_rows = []
        for model_name, metrics in test_res.items():
            test_rows.append({
                "Model": model_name,
                "Test MAE (kg/ha)": metrics.get("test_MAE", ""),
                "Test MSE": metrics.get("test_MSE", ""),
                "Test RMSE (kg/ha)": metrics.get("test_RMSE", ""),
                "Test R²": metrics.get("test_R2", ""),
            })
        test_df = pd.DataFrame(test_rows)
        st.dataframe(test_df, use_container_width=True)
    
    # Model selection
    selected = results.get("selected_model_name", "N/A")
    st.success(f"**Selected Model: {selected}**")
    for note in results.get("selection_notes", []):
        st.markdown(note)
    
    # Plots
    st.markdown("---")
    st.subheader("4. Evaluation Plots")
    
    col1, col2 = st.columns(2)
    with col1:
        avp = load_image("actual_vs_predicted.png")
        if avp:
            st.image(avp, caption="Actual vs Predicted (Validation)", use_container_width=True)
    with col2:
        resid = load_image("residual_comparison.png")
        if resid:
            st.image(resid, caption="Residual Distributions (Validation)", use_container_width=True)
    
    overfit = load_image("overfitting_check.png")
    if overfit:
        st.image(overfit, caption="Training vs Validation MAE (Overfitting Check)", 
                 use_container_width=True)
    
    # Feature importances
    fi = load_image("feature_importances.png")
    if fi:
        st.image(fi, caption="Feature Importances (CART & Random Forest)", use_container_width=True)


# ============================================================================
# PAGE 4: YIELD PREDICTION
# ============================================================================
elif page == "🔮 Predict Yield":
    st.title("🌾 Crop Yield Predictor")
    st.markdown("Enter growing conditions to get yield estimates from the trained models.")
    
    st.info(
        "**Note:** These predictions are statistical estimates based on the training data. "
        "Actual crop yield depends on many factors not captured in this model."
    )
    
    # Check if models exist
    try:
        from src.prediction import predict, load_models
        models = load_models()
        feature_ranges = models["feature_ranges"]
        feature_cols = models["feature_columns"]
    except FileNotFoundError as e:
        st.error(f"Models not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
    
    # Input fields
    st.subheader("Input Features")
    
    col1, col2 = st.columns(2)
    
    inputs = {}
    for i, col in enumerate(feature_cols):
        display = FEATURE_DISPLAY.get(col, col)
        fmin, fmax = feature_ranges.get(col, (0, 100))
        
        # Default to midpoint of training range
        default = round((fmin + fmax) / 2, 1)
        
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            st.markdown(f"**{display}**")
            st.caption(f"Training range: [{fmin:.1f}, {fmax:.1f}]")
            val = st.number_input(
                display,
                min_value=0.0,
                max_value=fmax * 2,
                value=default,
                step=0.1,
                key=f"input_{col}",
                label_visibility="collapsed",
            )
            inputs[col] = val
    
    # Predict button
    if st.button("🔮 Predict Yield", type="primary", use_container_width=True):
        try:
            result = predict(inputs, models)
            
            # Warnings
            if result.get("warnings"):
                for w in result["warnings"]:
                    st.warning(w)
            
            # Predictions
            st.markdown("---")
            st.subheader("Predictions")
            
            pred_cols = st.columns(len(result["predictions"]))
            for col, (model_name, pred_val) in zip(pred_cols, result["predictions"].items()):
                with col:
                    st.metric(model_name, f"{pred_val:,.0f} kg/ha")
            
            # Differences
            preds = list(result["predictions"].values())
            if len(preds) > 1:
                max_diff = max(preds) - min(preds)
                st.markdown(f"**Max difference between models:** {max_diff:,.0f} kg/ha")
            
            # Prediction interval
            if "prediction_interval" in result and "error" not in result["prediction_interval"]:
                pi = result["prediction_interval"]
                st.markdown("---")
                st.subheader("Regression Prediction Interval (95%)")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Lower Bound", f"{pi['pi_lower']:,.0f} kg/ha")
                col2.metric("Point Estimate", f"{pi['point_estimate']:,.0f} kg/ha")
                col3.metric("Upper Bound", f"{pi['pi_upper']:,.0f} kg/ha")
                
                st.markdown(
                    f"**95% Confidence Interval for mean:** "
                    f"[{pi['ci_lower']:,.0f}, {pi['ci_upper']:,.0f}] kg/ha"
                )
                st.markdown(
                    "The prediction interval is wider than the confidence interval because it "
                    "accounts for individual observation variability in addition to estimation uncertainty."
                )
        except Exception as e:
            st.error(f"Prediction error: {e}")


# ============================================================================
# PAGE 5: INSIGHTS
# ============================================================================
elif page == "💡 Insights":
    st.title("💡 Key Insights & Observations")
    
    if results is None:
        st.error("Training results not found. Run `python -m src.model_training` first.")
        st.stop()
    
    stat = results.get("statistical_analysis", {})
    
    # Key correlations
    st.subheader("1. Key Correlations")
    corr = stat.get("correlation", {})
    if corr.get("results"):
        for item in corr["results"]:
            emoji = "🔗" if abs(item["r"]) > 0.3 else "🔹"
            st.markdown(f"{emoji} **{item['display_name']}**: {item['interpretation']}")
    
    st.info(
        "**Reminder:** Correlation does not imply causation. These observations describe "
        "statistical associations in the data, not cause-and-effect relationships."
    )
    
    # Decision rules
    st.markdown("---")
    st.subheader("2. CART Decision Rules")
    st.markdown(
        "The CART tree captures non-linear patterns as interpretable IF-THEN rules. "
        "These rules describe patterns found in the training data."
    )
    
    rules_path = os.path.join(OUTPUT_DIR, "cart_rules.txt")
    if os.path.exists(rules_path):
        with open(rules_path) as f:
            rules = f.read()
        # Show first few rules for readability
        rule_lines = rules.strip().split("\n")
        display_rules = "\n".join(rule_lines[:30])
        if len(rule_lines) > 30:
            display_rules += f"\n... ({len(rule_lines) - 30} more lines)"
        st.code(display_rules, language="text")
    
    # What-if scenarios
    st.markdown("---")
    st.subheader("3. What-If Scenario Comparison")
    st.markdown("Compare predictions under different input scenarios.")
    
    try:
        from src.prediction import predict, load_models
        models = load_models()
        feature_cols = models["feature_columns"]
        feature_ranges = models["feature_ranges"]
        
        # Define scenarios based on training data ranges
        scenarios = {}
        for col in feature_cols:
            fmin, fmax = feature_ranges[col]
            mid = (fmin + fmax) / 2
            scenarios[col] = {
                "Low": round(fmin + (fmax - fmin) * 0.2, 1),
                "Medium": round(mid, 1),
                "High": round(fmin + (fmax - fmin) * 0.8, 1),
            }
        
        scenario_results = []
        for level in ["Low", "Medium", "High"]:
            inputs = {col: scenarios[col][level] for col in feature_cols}
            result = predict(inputs, models)
            row = {"Scenario": f"{level} inputs"}
            for col in feature_cols:
                row[FEATURE_DISPLAY.get(col, col)] = inputs[col]
            for model, pred in result["predictions"].items():
                row[model] = f"{pred:,.0f}"
            scenario_results.append(row)
        
        st.dataframe(pd.DataFrame(scenario_results), use_container_width=True)
        
        st.markdown(
            "**Note:** These scenarios use the 20th, 50th, and 80th percentile of each "
            "feature's training range simultaneously. In reality, features may not vary "
            "independently."
        )
    except Exception as e:
        st.warning(f"Could not generate scenarios: {e}")
    
    # Model comparison insights
    st.markdown("---")
    st.subheader("4. Model Selection & Interpretability")
    
    selected = results.get("selected_model_name", "N/A")
    best = results.get("best_model_name", "N/A")
    
    st.markdown(f"**Best model by validation MAE:** {best}")
    st.markdown(f"**Selected model:** {selected}")
    
    for note in results.get("selection_notes", []):
        st.markdown(note)
    
    st.markdown("""
    **Interpretability vs Accuracy Trade-off:**
    - **Linear Regression:** Most interpretable (coefficients, p-values, CIs), but assumes linearity.
    - **CART Tree:** Moderate interpretability (IF-THEN rules, visual tree), captures non-linearity.
    - **Random Forest:** Least interpretable (black-box ensemble), often best accuracy.
    
    For a domain like agriculture, interpretable models are often preferred because 
    stakeholders need to understand *why* a prediction was made, not just *what* it is.
    """)
    
    # Limitations
    st.markdown("---")
    st.subheader("5. Limitations & Disclaimers")
    st.markdown(f"""
    - **Data Source:** {'Real agricultural dataset' if DATA_SOURCE == 'REAL' else 'Simulated data'}.
    - Only 4 features are used; real crop yield depends on many more factors 
      (soil nutrients, pests, irrigation, crop variety, etc.).
    - Models capture patterns in the training data and may not generalise to 
      different regions, climates, or crop types.
    - Statistical significance (p < 0.05) does not imply practical significance 
      or causal relationships.
    - Predictions should be treated as rough estimates, not prescriptions.
    - Predictions are estimates and should not be treated as agronomic recommendations.
    """)
