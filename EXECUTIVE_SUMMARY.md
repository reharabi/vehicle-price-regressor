# Executive Summary — Car Price Prediction

---

## The Problem

Pricing a used car is surprisingly difficult. Two identical cars — same make, model, and year — can differ by thousands of dollars based on mileage, condition, colour, and market demand. Dealerships and private sellers frequently misprice inventory: underpricing costs revenue; overpricing drives buyers away.

This project builds a **machine learning model that predicts used car prices automatically** from a car's observable characteristics — giving dealerships, buyers, and sellers a data-driven benchmark for any listing.

---

## What Was Built

A complete data pipeline that:

1. **Loads and cleans raw listings data** — removes corrupt entries, fixes data types, standardises text
2. **Explores the data** — identifies which characteristics most strongly drive price
3. **Engineers meaningful features** — transforms raw year and mileage into more predictive signals
4. **Prepares data for modelling** — handles missing values, scales numbers, encodes categories
5. **Trains and compares four models** — selecting the best performer on unseen data
6. **Generates interpretable insights** — explains what the model learned and why

---

## The Result

| Model | Typical Error (MAE) | Max Error Sensitivity (RMSE) |
|---|---|---|
| **XGBoost** ← recommended | **$1,219** | **$2,551** |
| Random Forest | $1,539 | $3,382 |
| Decision Tree | $2,153 | $4,368 |
| Linear Regression | $2,607 | $5,235 |

**XGBoost is the recommended model.** Across 11,227 unseen test listings, it predicted car prices within **$1,219 on average** — with an RMSE of $2,551 meaning it is well-controlled even on hard cases like luxury vehicles.

All four models passed the overfitting check — trained and test performance are consistent, confirming the models have learned genuine patterns rather than memorising the training data.

---

## What Drives Car Prices

### The Three Strongest Signals

**1. Car Age — the dominant factor**
The Decision Tree's first split alone separates cars into two groups with a **$10,810 average price gap**:
- Cars ≤ 3.5 years old: average **$15,454**
- Cars > 3.5 years old: average **$4,644**

Every additional year of age costs approximately **$312** in predicted price.

**2. Brand — a $10,000+ premium**
Premium brands command significantly higher prices than economy brands:
- Porsche badge: **+$10,533** compared to a similar car without it
- Electric vehicles: **+$22,837** premium — the single largest price factor in the linear model
- Economy brands (Daewoo, Opel, Lada): priced at a significant discount

**3. Engine Size — separates luxury from standard within newer cars**
Among newer cars (≤ 3.5 years), engine size creates the biggest price split of all:
- Smaller engine newer cars: average $13,084
- Larger engine newer cars: average **$30,496**
- Gap: **$17,412** — more than any other single factor

---

## Other Key Findings

| Factor | Effect on Price |
|---|---|
| **Mileage** | Clear negative relationship — each additional 80,000 km costs $1,362 in predicted price |
| **Transmission** | Automatic cars priced 15–20% higher than manual equivalent |
| **Drive unit** | AWD / 4WD vehicles command a premium over front-wheel drive |
| **Fuel type** | Diesel cars typically priced higher than petrol equivalents |
| **Condition** | Cars listed 'with mileage' (normal use) priced higher than damaged or for-parts |
| **Segment** | Higher segments (E, F, S) consistently command premium prices over economy segments (A, B) |

---

## How It Works in Practice

Once deployed, the model can be used as a **pricing assistant**:

1. Enter a car's characteristics (age, mileage, brand, fuel type, etc.)
2. The model returns a predicted price in seconds
3. The dealership compares the predicted price to the actual listing:
   - **Listed well below predicted** → underpriced — opportunity to buy
   - **Listed well above predicted** → overpriced — negotiate or avoid
   - **Listed near predicted** → fairly priced — proceed with confidence

At XGBoost's MAE of $1,219, the model provides a reliable benchmark for **95% of the market** (cars priced under $50,000). Luxury vehicles at the top end of the market have naturally higher prediction errors due to their rarity in the training data.

---

## Business Applications

| Use Case | How the Model Helps |
|---|---|
| **Inventory pricing** | Set asking prices based on data, not intuition — reduce days-on-lot |
| **Auction bidding** | Instant price estimate before bidding — avoid overbidding on overpriced vehicles |
| **Market monitoring** | Identify underpriced listings in the market before competitors do |
| **Trade-in valuation** | Provide customers with an objective, explainable trade-in benchmark |
| **Depreciation planning** | Quantify how age and mileage will affect resale value over time |

---

## Limitations

| Limitation | Impact | Potential Fix |
|---|---|---|
| Right-skewed prices | Higher errors on luxury cars ($50k+) | Log-transform target variable; retrain |
| No hyperparameter tuning | XGBoost used default settings — room to improve | Add Optuna/GridSearchCV tuning step |
| Eastern European market | Model may not generalise to other regions | Retrain on local market data |
| No condition detail | Interior quality, service history not captured | Enrich dataset with additional features |
| Static model | Prices change with market conditions | Retrain periodically on fresh listings |

---

## Recommended Next Steps

1. **Deploy XGBoost** — the model is production-ready for the current market dataset (MAE $1,219, no overfitting)
2. **Add hyperparameter tuning** — a cross-validated RMSE optimisation sweep on `n_estimators`, `learning_rate`, and `max_depth` would likely reduce RMSE below $2,000
3. **Test log-price transformation** — normalising the target distribution could improve performance on luxury vehicles
4. **Retrain on fresh data** — car market prices shift with fuel costs, economic conditions, and new model releases; quarterly retraining is recommended
5. **Build a prediction interface** — wrap the model in a simple web form (Streamlit or Flask) where a dealership employee can enter a car's details and get an instant price estimate

---

## Summary

This project demonstrates that a structured machine learning pipeline — applied to 56,000 used car listings — can predict prices within $1,219 on average with no overfitting. The XGBoost model is robust, interpretable, and ready for practical use. Three factors dominate pricing: **car age, brand, and engine size** — findings that align with common market intuition and are validated independently by every model in the pipeline.
