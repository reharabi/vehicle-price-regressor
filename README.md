# Car Price Prediction

**Automotive | Machine Learning | Regression**

A machine learning pipeline that predicts used car prices in USD — built end-to-end from raw data to final model evaluation.

---

## Results at a Glance

| Model | Test MAE | Test RMSE | Train-Test Gap | Overfit |
|---|---|---|---|---|
| **XGBoost** ← winner | **$1,219** | **$2,551** | OK | No |
| Random Forest | $1,539 | $3,382 | OK | No |
| Decision Tree | $2,153 | $4,368 | OK | No |
| Linear Regression | $2,607 | $5,235 | OK | No |

**XGBoost predicts car prices within $1,219 on average** — with no overfitting and the lowest RMSE across all models.

---

## The Business Problem

Used car pricing is inconsistent. Dealerships, private sellers, and buyers all face the same challenge — is this car priced fairly? A data-driven model that predicts price from a car's characteristics allows:

- Dealerships to set competitive, evidence-based prices
- Buyers to identify underpriced vehicles in the market
- Auction participants to make faster, more informed bidding decisions

---

## Project Structure

## 📂 Repository Structure & Navigation

- 📓 **[Jupyter Notebook](https://github.com/reharabi/vehicle-price-regressor/blob/main/car_price_pred.ipynb)** — Full end-to-end Python pipeline notebook (Google Colab)
- 📊 **[Executive Summary](EXECUTIVE_SUMMARY.md)** — Business-focused findings, price drivers, and recommendations
- 🔬 **[Technical Analysis](TECHNICAL_ANALYSIS.md)** — Full engineering methodology, preprocessing, and modeling decisions
- 📋 **[cars.csv](cars.csv)** — Project dataset *(Note: Place this in your Google Drive at `MyDrive/cars.csv` before running the notebook)*
- 📝 **[README.md](README.md)** — This project overview file

---

## Pipeline Overview

```
Raw CSV (cars.csv)
    ↓
Step 2 — Data Gathering      (load, inspect shape, types, value counts)
    ↓
Step 3 — Data Cleaning       (remove corrupt rows, fix types, lowercase text)
    ↓
Step 4 — EDA                 (price distribution, brand trends, depreciation, correlation)
    ↓
Step 5 — Feature Engineering (car_age, mileage_per_year, make_category, price_category)
    ↓
Step 6 — Preprocessing       (stratified split, imputation, scaling, one-hot encoding)
    ↓
Step 7 — Model Training      (Linear Regression, Random Forest, XGBoost, Decision Tree)
    ↓
Step 8 — Prediction & Evaluation (scatter plots, residuals, feature importance, best model)
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Used car listings — Eastern European market |
| Raw size | 56,244 rows |
| After cleaning | 56,133 rows (111 corrupt rows removed) |
| Features | 12 raw → 70 after engineering + encoding |
| Target | `priceUSD` — listing price in US dollars |
| Price range | $100 – $235,235 |
| Split | 80% train (44,906) / 20% test (11,227) — stratified |

---

## Key Design Decisions

**Why RMSE as primary metric?**
Car prices are heavily right-skewed — a small number of luxury cars have very high values. A $20,000 error on a luxury car is far worse than twenty $1,000 errors on economy cars. RMSE squares the errors before averaging, so large mistakes are penalised more heavily. MAE treats all errors equally and is reported as a secondary, dollar-interpretable metric.

**Why XGBoost wins?**
XGBoost builds trees sequentially — each new tree corrects the errors of the previous ones. This boosting mechanism efficiently targets the hardest-to-predict cars (often luxury vehicles or unusual configurations). After 100 rounds, the ensemble converges to the most accurate predictions.

**Why different feature counts per model type?**
All models use the same 70 preprocessed features. Feature importance charts show that `car_age`, `make_category`, and `mileage_kilometers` dominate across both XGBoost and Random Forest — confirming that the engineered features carry the most signal.

**Why sklearn Pipelines are used end-to-end?**
All scalers and encoders are fit only on training data and applied to test data. This prevents data leakage — the model never sees test data statistics during training, ensuring honest evaluation.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Language |
| pandas, numpy | Data handling |
| scikit-learn | Preprocessing, LR, RF, DT, metrics |
| XGBoost | Gradient boosting regressor |
| matplotlib, seaborn | Visualisation |
| Google Colab | Notebook environment |
| Google Drive | Data storage |

---

## How to Run

1. Upload `cars.csv` to your Google Drive at `MyDrive/cars.csv`
2. Open `car_price_prediction.ipynb` in Google Colab
3. Run **Runtime → Run all**

All libraries are installed in the first cell (`!pip install xgboost`).

---

## What the Notebook Covers

1. **Data Gathering** — load CSV, inspect shape, dtypes, value counts, unique values
2. **Data Cleaning** — remove placeholder mileage rows, corrupt prices, duplicates; report outliers
3. **EDA** — cardinality analysis, price distribution, brand price comparison, depreciation curves, correlation heatmap, categorical feature effects
4. **Feature Engineering** — create `car_age`, `mileage_per_year`, `make_category`; bin prices for stratified split; drop redundant columns
5. **Preprocessing** — stratified 80/20 split, imputation (median/mode/0), StandardScaler, OneHotEncoder
6. **Model Training** — four models trained and evaluated with MAE, RMSE, R²
7. **Prediction & Evaluation** — actual vs predicted scatter plots, residual plots, MAE/RMSE bar charts, LR coefficients, feature importances, Decision Tree visualisation, XGBoost boosting curve

---

## Limitations & Next Steps

- **Right-skewed prices** — predictions for luxury cars ($50,000+) have higher errors; log-transforming the target is a potential improvement
- **No hyperparameter tuning** — XGBoost used default learning rate and depth; GridSearchCV or Optuna could improve performance further
- **Market-specific dataset** — listings are from an Eastern European market; model may not generalise to other regions without retraining
- **Next step** — add a hyperparameter tuning section using cross-validated RMSE as the optimisation objective
