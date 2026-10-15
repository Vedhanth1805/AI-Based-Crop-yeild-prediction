# Crop Yield Prediction

An interactive Streamlit application that estimates rice crop yield from rainfall, temperature, soil pH, and fertilizer inputs. It includes saved regression and tree-based models, data analysis, model comparisons, and prediction intervals.

---

## 📌 Project Overview

This project implements a statistical analysis and predictive modeling pipeline to estimate crop yield ($\text{kg/hectare}$) using key environmental and agricultural inputs:
- **Rainfall** ($\text{mm}$)
- **Temperature** ($^\circ\text{C}$)
- **Soil pH**
- **Fertilizer Amount** ($\text{kg/hectare}$)

The project includes hypothesis tests, confidence and prediction intervals, ANOVA, OLS regression, CART decision trees with pruning, and Random Forest models.

---

## 🗂️ Project Directory Structure

```text
crop-yield-prediction/
├── app.py                      # 5-page interactive Streamlit dashboard
├── requirements.txt            # Pinned dependencies
├── README.md                   # Project documentation and user guide
├── PROJECT_REVIEW.md           # Technical project review
├── data/
│   ├── crop_data.csv           # Dataset used by the app
│   └── crop_data_candidate.csv # Alternate candidate dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data validation, leakage prevention, 70/15/15 split
│   ├── statistical_analysis.py # Correlation, ANOVA, OLS regression, diagnostics
│   ├── model_training.py       # Model training, tuning, pruning, plots
│   ├── evaluation.py           # Metrics (MAE, MSE, RMSE, R²), comparison tables
│   └── prediction.py           # Inference engine with 95% prediction intervals
├── models/                     # Serialized artifacts (.joblib)
│   ├── linear_regression.joblib
│   ├── cart_tree.joblib
│   ├── random_forest.joblib
│   ├── ols_statsmodels.joblib
│   ├── preprocessing_pipeline.joblib
│   ├── feature_columns.joblib
│   └── feature_ranges.joblib
└── outputs/                    # Generated charts, tables, rules, and metrics JSON
    ├── training_results.json
    ├── statistical_analysis.json
    ├── cart_rules.txt
    ├── correlation_heatmap.png
    ├── scatter_plots.png
    ├── anova_boxplot_soil_type.png
    ├── anova_boxplot_fertilizer_level.png
    ├── residual_plots.png
    ├── cart_tree.png
    ├── feature_importances.png
    ├── actual_vs_predicted.png
    ├── residual_comparison.png
    └── overfitting_check.png
```

---

## Project methods

- Pearson correlation and regression analysis
- One-way ANOVA for group comparisons
- Train/validation/test splitting and preprocessing pipelines
- Linear Regression, CART Regression Tree, and Random Forest
- Model comparison using MAE, MSE, RMSE, and $R^2$
- CART decision rules and prediction intervals

---

## ⚙️ Setup & Installation Instructions

### 1. Clone / Navigate to Directory
```bash
git clone https://github.com/Vedhanth1805/AI-Based-Crop-yeild-production.git
cd AI-Based-Crop-yeild-production
```

### 2. Create and Activate Virtual Environment

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Open the prediction app

From the project folder, activate the environment and start the app:
```bash
source venv/bin/activate
streamlit run app.py
```

The app opens in your browser. If it does not, visit `http://localhost:8501`. If that port is already in use, Streamlit will show another local URL, such as `http://localhost:8502`.

Choose **Predict Yield**, enter rainfall, temperature, soil pH, and fertilizer values, then click **Predict Yield**. The saved models load automatically; you do not need to retrain them for normal use. To stop the app, press **Ctrl+C** in the Terminal running it.

### Optional: Retrain the models

Only run this if you want to rebuild the saved models and analysis outputs:
```bash
python -m src.model_training
```

---

## 📊 Dataset & Column Adaptation

- **Default Dataset Location:** `data/crop_data.csv`
- **Data Source:** Real agricultural crop yield benchmark dataset (10,000 total records across 6 crops; subset to 1,702 **Rice** records to avoid crop-type variance masking environmental factor effects).
- **Target Variable:** Crop Yield in $\text{kg/hectare}$ (converted from metric tons/hectare $\times 1000$).

### Dropping in a Custom Dataset:
If you have a different CSV, place it at `data/crop_data.csv` and edit the column dictionary at the top of `src/data_preprocessing.py`:
```python
COLUMN_MAP = {
    "rainfall": "Your_Rainfall_Column_Name",
    "temperature": "Your_Temperature_Column_Name",
    "soil_ph": "Your_Soil_pH_Column_Name",
    "fertilizer": "Your_Fertilizer_Column_Name",
    "yield": "Your_Yield_Column_Name",
    "soil_type": "Your_Soil_Type_Column_Name",  # optional for ANOVA
    "crop_type": "Your_Crop_Type_Column_Name",  # optional
}
```
Set `CROP_FILTER = None` if your dataset is already single-crop or pre-cleaned.

---

## 📈 Actual Experimental Results (Rice Crop Yield)

### 1. Pearson Correlation Analysis
| Feature | Pearson $r$ | $p$-value | Significant ($\alpha=0.05$) | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Fertilizer (kg/ha)** | **+0.5291** | $7.00 \times 10^{-87}$ | **Yes** | Strong positive linear correlation |
| **Rainfall (mm)** | **+0.4616** | $6.75 \times 10^{-64}$ | **Yes** | Moderate positive linear correlation |
| **Soil pH** | **-0.0348** | $0.230$ | No | Negligible linear correlation |
| **Temperature ($^\circ\text{C}$)** | **-0.0292** | $0.314$ | No | Negligible linear correlation |

### 2. OLS Multiple Linear Regression
- **Fitted Model:**  
  $$\text{Yield} = 6123.33 + 1.004 \cdot \text{Rainfall} + 0.259 \cdot \text{Temperature} - 55.361 \cdot \text{Soil pH} + 9.885 \cdot \text{Fertilizer}$$
- **Model Fit:** $R^2 = 0.4673$, $\text{Adjusted } R^2 = 0.4655$
- **Overall $F$-test:** $F(4, 1186) = 260.10$, $p = 1.77 \times 10^{-160}$ (Highly Significant)
- **Multicollinearity:** Variance Inflation Factors (VIF): Rainfall ($5.36$), Temperature ($8.24$), Soil pH ($14.70$), Fertilizer ($6.56$).

### 3. One-Way ANOVA
- **Fertilizer Tertiles (Low / Medium / High):**  
  $F(2, 1188) = 204.52$, $p = 4.70 \times 10^{-77}$ $\rightarrow$ **Reject $H_0$** (Mean yield increases significantly from 8,299 kg/ha to 10,319 kg/ha).
- **Soil Types (Loamy, Sandy, Silt, Clay):**  
  $F(3, 1187) = 0.3005$, $p = 0.825$ $\rightarrow$ **Fail to reject $H_0$** (No significant yield difference across soil textures alone in this dataset).

### 4. Model Comparison

| Metric | Multiple Linear Regression | CART Tree | Random Forest |
| :--- | :---: | :---: | :---: |
| **Training MAE (kg/ha)** | 957.08 | 871.73 | 674.47 |
| **Training $R^2$** | 0.4673 | 0.5651 | 0.7290 |
| **Validation MAE (kg/ha)** | 1025.43 | **1004.06** | **963.45** |
| **Validation $R^2$** | 0.4408 | 0.4564 | 0.5010 |
| **5-Fold CV MAE (kg/ha)** | $961.15 \pm 17.83$ | $1029.24 \pm 39.51$ | $968.48 \pm 16.31$ |
| **Test Set MAE (kg/ha)** | 990.42 | 1020.77 | 972.04 |
| **Test Set $R^2$** | 0.4741 | 0.4385 | 0.4913 |
| **Interpretability** | High (Coefficients, $t$-tests) | **High (Visual tree, IF-THEN rules)** | Low (Black-box ensemble) |

**Winning Model Selection:**
CART Regression Tree is chosen as the **primary defended model** because its validation error is within **$4.2\%$** of the Random Forest ensemble, while providing complete transparency through visual nodes and induced decision rules. Random Forest is presented as an ensemble extension demonstrating variance reduction.

---

## ⚠️ Limitations

1. **Domain Scope:** Predictions are statistical estimates and are **not** agricultural fertilizer prescriptions.
2. **Correlation vs. Causation:** Significant correlation between fertilizer/rainfall and yield indicates association, not pure direct causation.
3. **Feature Boundaries:** The model only uses 4 environmental variables. Real agricultural production depends on seed variety, pest pressure, micronutrients (NPK), solar radiation, and irrigation scheduling.
