# Crop Yield Prediction — Technical Overview

## Project summary

This project is a Streamlit application for exploring crop-yield data and estimating rice yield from rainfall, temperature, soil pH, and fertilizer input. It includes a saved-model prediction workflow, statistical analysis, model comparison, and visual summaries.

## Data

The application reads `data/crop_data.csv`, checks required columns, filters the rows to Rice, and converts crop yield from tonnes per hectare to kilograms per hectare. The resulting app dataset contains 1,702 rice records. The prediction models use four numeric inputs: rainfall, temperature, soil pH, and fertilizer amount.

The repository also includes `data/crop_data_candidate.csv`, an alternate candidate dataset. The active app data path is configured in `src/data_preprocessing.py`.

## Application workflow

1. Load and validate the CSV data.
2. Filter to the selected crop and standardize column names.
3. Split data into training, validation, and test subsets.
4. Fit preprocessing on training data to reduce leakage.
5. Train and compare Linear Regression, CART Regression Tree, and Random Forest models.
6. Save trained estimators and evaluation outputs under `models/` and `outputs/`.
7. Load the saved models in Streamlit to generate predictions and prediction intervals.

## Models and evaluation

- **Linear Regression:** interpretable linear baseline and basis for interval estimates.
- **CART Regression Tree:** tree-based model with readable decision rules.
- **Random Forest:** ensemble of trees that can capture more complex patterns.

The project reports MAE, MSE, RMSE, and R², along with diagnostic plots and statistical analyses. The included results show CART test MAE of approximately 1,021 kg/ha and test R² of approximately 0.44. Validation and test scores are reported separately; validation results are used during model selection, while the test set is reserved for final evaluation.

## Running the app

From the project directory on macOS or Linux:

```bash
source venv/bin/activate
streamlit run app.py
```

For a fresh setup, create a virtual environment and install dependencies from `requirements.txt`. Open the local URL printed by Streamlit. On the **Predict Yield** page, enter the four input values and select **Predict Yield**. The saved model files are loaded automatically; retraining is not needed for normal predictions.

To regenerate the models and analysis outputs, run:

```bash
python -m src.model_training
```

## Limitations

- Only four numeric features are used; actual yields also depend on crop variety, pests, irrigation, nutrients, weather over time, and local conditions.
- The data reflects a particular dataset and crop subset; model quality may not generalize to other crops, locations, or seasons.
- Statistical association does not establish causation.
- Predictions are approximate estimates, not guaranteed yields or farming prescriptions.
- The dataset’s original source and redistribution license should be verified before reuse outside this repository.
