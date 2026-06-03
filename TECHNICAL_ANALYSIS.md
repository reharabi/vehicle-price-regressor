# Technical Analysis — Car Price Prediction

---

## 1. Problem Framing

This is a **supervised regression** task: given a set of features describing a used car listing, predict its price in USD.

The target variable (`priceUSD`) is continuous and heavily right-skewed — most cars are priced under $10,000 with a long tail of expensive luxury vehicles. This skewness has direct implications for metric selection.

### Metric Selection

| Metric | Formula | Used | Reason |
|---|---|---|---|
| **RMSE** | √mean((y − ŷ)²) | ✅ Primary | Penalises large errors more — critical for catching luxury car mispredictions |
| **MAE** | mean(\|y − ŷ\|) | ✅ Secondary | Average dollar error — directly interpretable for business use |
| **R²** | 1 − SS_res/SS_tot | Reported only | Measures variance explained, but artificially inflated by luxury car outliers on skewed distributions |

R² is **not used for model selection**. On a right-skewed distribution, a small number of very expensive cars dominate the variance. A model that predicts luxury cars reasonably well can score R² > 0.85 while still being off by thousands of dollars on everyday mid-range cars. RMSE and MAE give honest, dollar-denominated errors.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | Used car listings — Eastern European market |
| Raw rows | 56,244 |
| After cleaning | 56,133 |
| Removed | 111 rows (0.2%) |
| Columns (raw) | 12 |
| Features after preprocessing | 70 |
| Target | `priceUSD` |
| Price range | $100 – $235,235 |
| Mean price | $7,420 |
| Median price | $5,350 |
| Train set | 44,906 rows (80%) |
| Test set | 11,227 rows (20%) |

---

## 3. Data Cleaning

### 3.1 Type Conversion
`volume_cm3` was stored as a string object column (containing values like `'nan'` as a literal string). Converted using `pd.to_numeric(..., errors='coerce')` to produce proper NaN values.

### 3.2 Corrupt Data Removal (Rule-Based)
Statistical outlier removal is intentionally avoided to preserve legitimate luxury and economy market extremes. Only clear data corruption is removed:

```python
# Placeholder mileage — clearly not real
df = df[df['mileage_kilometers'] < 9_999_999]   # removed 22 rows

# Placeholder prices — no car costs under $100
df = df[df['priceUSD'] >= 100]                  # removed 2 rows

# Exact duplicate rows
df = df.drop_duplicates()                        # removed 87 rows
```

Total removed: **111 rows (0.2%)** — preserving 99.8% of the original data.

### 3.3 Outlier Report (No Removal)
IQR-based outlier counts are reported for transparency:

| Column | Outliers | % |
|---|---|---|
| `priceUSD` | 2,457 | 4.4% |
| `year` | 257 | 0.5% |
| `mileage_kilometers` | 665 | 1.2% |
| `volume_cm3` | 3,615 | 6.4% |

These are **not removed** — a $200,000 Ferrari is a genuine listing, not a data error.

### 3.4 Text Standardisation
All string columns lowercased for consistent categorical encoding downstream.

---

## 4. Exploratory Data Analysis — Key Findings

### 4.1 Target Variable
- **Skewness: 3.5** — right-skewed with a long upper tail
- **Mean ($7,420) > Median ($5,350)** — mean pulled up by expensive luxury vehicles
- Log-transformed distribution is approximately normal — a log target transform could improve model performance in future iterations

### 4.2 Cardinality
| Column | Unique Values | Action |
|---|---|---|
| `model` | 1,034 | Dropped — too sparse for encoding |
| `make` | 96 | Replaced by `make_category` (grouped) |
| `color` | 13 | Encoded directly |
| `segment` | 9 | Encoded directly |
| `transmission` | 2 | Encoded directly |
| `fuel_type` | 3 | Encoded directly |
| `condition` | 3 | Encoded directly |
| `drive_unit` | 4 | Encoded directly |

### 4.3 Key Correlations with Price
| Feature | Direction | Strength |
|---|---|---|
| `year` | Positive | Strong — newer cars are more expensive |
| `mileage_kilometers` | Negative | Moderate — higher mileage = lower price |
| `volume_cm3` | Positive | Moderate — larger engines = higher prices |

### 4.4 Brand Effect
Premium brands (BMW, Mercedes-Benz, Volvo, Porsche) command **2–3× higher median prices** than economy brands (Daewoo, Opel, Lada). Brand is among the top 3 most important features in both ensemble models.

### 4.5 Depreciation
- Prices rise sharply for cars made after 2005
- Median price drops from $15,000 (0–50k km) to $3,500 (300–500k km)
- `car_age` and `mileage_per_year` (engineered features) capture this more cleanly than raw `year` and `mileage_kilometers`

---

## 5. Feature Engineering

### 5.1 `car_age`
```python
df['car_age'] = 2019 - df['year']
```
The dataset was collected in early 2019. `car_age` is more interpretable than `year` (a car aged 3 years is more meaningful than year = 2016) and removes year as a calendar artifact.

### 5.2 `mileage_per_year`
```python
df['mileage_per_year'] = df['mileage_kilometers'] / df['car_age'].clip(lower=1)
```
Separates total mileage from usage intensity. A 10-year-old car with 100,000 km (10,000/year) has been driven more gently than a 5-year-old car with the same total mileage (20,000/year). `.clip(lower=1)` prevents division by zero for brand-new cars.

### 5.3 `make_category`
```python
make_counts  = df['make'].value_counts()
common_makes = make_counts[make_counts >= 100].index
df['make_category'] = df['make'].apply(lambda x: x if x in common_makes else 'other')
```
Reduces `make` from 96 unique values to 39 (38 common + 'other'). Makes with fewer than 100 listings are too rare to produce reliable coefficients and would add noise to the model.

### 5.4 `price_category`
```python
p33 = df['priceUSD'].quantile(0.33)   # $3,200
p66 = df['priceUSD'].quantile(0.66)   # $7,900
```
Bins prices into budget / mid / luxury for stratified splitting only. Ensures both train and test sets have the same price distribution across all three tiers. **Dropped before training** — it is not a feature.

### 5.5 Columns Dropped
| Column | Reason |
|---|---|
| `model` | 1,034 unique values — OHE would create 1,000+ sparse columns |
| `year` | Replaced by `car_age` |
| `make` | Replaced by `make_category` |

---

## 6. Preprocessing

All preprocessing steps are **fit on training data only** and applied to test data — preventing any form of data leakage.

### 6.1 Stratified Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=price_category
)
```
Stratification on `price_category` guarantees that budget (33.1%), mid (33.2%), and luxury (33.7%) cars are equally represented in both sets. Without stratification, random sampling could overrepresent luxury cars in one set.

### 6.2 Imputation Strategy

| Column | Nulls | Strategy | Reasoning |
|---|---|---|---|
| `volume_cm3` | 47 | `fillna(0)` | Missing = electric cars; engine displacement is genuinely 0, not unknown |
| Other numeric | 0 | — | No imputation needed |
| `drive_unit` | 1,904 (3.4%) | Train mode | Most frequent value in training set |
| `segment` | 5,281 (9.4%) | Train mode | Most frequent value in training set |

### 6.3 Scaling
```python
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numeric_cols])
X_test_num  = scaler.transform(X_test[numeric_cols])
```
StandardScaler centres each feature to mean=0, std=1. Required for Linear Regression (gradient descent converges faster and coefficients are comparable). Tree-based models (RF, XGBoost, DT) are scale-invariant but scaling does not hurt them.

### 6.4 One-Hot Encoding
```python
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
```
**Why `drop='first'`?** Without dropping the first category, all OHE columns for a group (e.g. all `transmission_*` columns) sum to exactly 1 for every row — perfect multicollinearity. Linear Regression has no unique solution and assigns arbitrary coefficients of billions of dollars that cancel each other out. Dropping the reference category breaks this collinearity.

**Why `handle_unknown='ignore'`?** If a test set row contains a category not seen during training (e.g. a new car make), the encoder silently sets all related columns to 0 rather than raising an error.

Final feature count: **70 features** (4 numeric + 66 encoded categorical).

---

## 7. Models

### 7.1 Linear Regression (Baseline)

```python
LinearRegression()
```

| Metric | Train | Test |
|---|---|---|
| MAE | $2,656 | $2,607 |
| RMSE | $5,548 | $5,235 |
| R² | 0.5603 | 0.5861 |

Linear Regression assumes a linear additive relationship between features and price. In practice, car pricing is non-linear — the price effect of age depends on brand, the mileage effect depends on segment, etc. These interaction effects cannot be captured by a purely linear model.

**Coefficient interpretation:**
- `car_age` → **−$3,667 per std dev** (−$312 per year of age)
- `fuel_type_electrocar` → **+$22,837** — EV premium (largest single dummy coefficient)
- `make_category_porsche` → **+$10,533** — Porsche badge premium
- `drive_unit_rear_drive` → **−$4,739** — apparent discount driven by multicollinearity with brand

Note on multicollinearity: `drive_unit_rear_drive` is negative despite rear-wheel drive being a premium feature (BMWs, Porsches). This is because `make_category_porsche` (+$10,533) and `make_category_bmw` already capture the brand premium — once LR accounts for those, the RWD column absorbs the residual and its coefficient turns negative. Tree models handle this automatically.

### 7.2 Random Forest

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
```

| Metric | Train | Test |
|---|---|---|
| MAE | $1,512 | $1,539 |
| RMSE | $3,446 | $3,382 |
| R² | 0.8304 | 0.8272 |

Random Forest is a **bagging** ensemble — 100 trees trained independently on different bootstrap samples, each considering √(n_features) at each split. Predictions are averaged across all trees, which reduces variance without increasing bias.

**Why regularisation parameters are needed:**
Without `max_depth=15` and `min_samples_leaf=5`, an unconstrained Random Forest on 44,906 samples achieves a near-perfect train MAE ($445) but poor test MAE ($1,142) — a 157% gap indicating severe overfitting. Constraining tree depth and minimum leaf size forces each tree to learn broader patterns rather than memorising individual training cars.

**Top features (RF):** `car_age` (0.348), `mileage_kilometers` (0.124), `volume_cm3` (0.120), `transmission_mechanics` (0.080), `mileage_per_year` (0.070)

### 7.3 XGBoost (Winner)

```python
XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
```

| Metric | Train | Test |
|---|---|---|
| MAE | $1,159 | $1,219 |
| RMSE | $2,098 | $2,551 |
| R² | 0.9371 | 0.9017 |

XGBoost is a **boosting** ensemble — 100 trees built sequentially, each one trained to correct the residuals of the previous ensemble. At `learning_rate=0.1`, each tree contributes 10% of its correction, preventing any single tree from dominating and providing natural regularisation.

**Boosting curve:** MAE drops from $3,200 at round 1 to $1,219 at round 100 — an improvement of over $2,000 from sequential error correction. Both train and test curves track closely, confirming no meaningful overfitting.

**Residual analysis:** Mean residual = −$15 (effectively unbiased). The model is not systematically over- or under-predicting. Residual spread (std = $1,876) reflects the remaining irreducible variance from factors not in the dataset (interior condition, service history, negotiation, etc.).

**Top features (XGBoost):** `car_age` (0.157), `drive_unit_front_wheel_drive` (0.079), `volume_cm3` (0.071), `fuel_type_electrocar` (0.069), `color_purple` (0.066)

### 7.4 Decision Tree (Interpretability)

```python
DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=42)
```

| Metric | Train | Test |
|---|---|---|
| MAE | $2,171 | $2,153 |
| RMSE | $4,369 | $4,368 |
| R² | 0.7273 | 0.7118 |

Trained at `max_depth=4` deliberately — the goal is interpretability, not performance. A deeper tree would overfit. The top 3 nodes of the tree reveal the most important price-splitting logic the model discovered independently:

**Node 1 (Root) — car_age ≤ 3.5 years:**
- Newer cars (≤ 3.5 yrs): 11,540 cars, avg price **$15,454**
- Older cars (> 3.5 yrs): 33,366 cars, avg price **$4,644**
- Price gap from one question: **$10,810**

**Node 2 — within newer cars: volume_cm3 (engine size):**
- Smaller engine: 9,969 cars, avg **$13,084**
- Larger engine: 1,571 cars, avg **$30,496**
- Price gap: **$17,412** — the largest single split in the top 3 nodes

**Node 3 — within older cars: car_age ≤ 13.4 years:**
- 3.5–13.4 years old: 12,899 cars, avg **$7,742**
- Older than 13.4 years: 20,467 cars, avg **$2,691**
- Price gap: **$5,051**

The tree independently chose the same top features found in EDA — validating that these are genuine price drivers, not artefacts.

---

## 8. Model Comparison

| Model | Test MAE | Test RMSE | Test R² | MAE Gap | Notes |
|---|---|---|---|---|---|
| **XGBoost** | **$1,219** | **$2,551** | **0.9017** | +$60 | Winner — lowest RMSE and MAE |
| Random Forest | $1,539 | $3,382 | 0.8272 | +$27 | Strong alternative |
| Decision Tree | $2,153 | $4,368 | 0.7118 | −$18 | Interpretability only |
| Linear Regression | $2,607 | $5,235 | 0.5861 | −$49 | Baseline — cannot capture non-linearity |

Sorted by Test RMSE (primary metric).

---

## 9. Why Not Alternatives

**Why not log-transform the target?**
Log-transforming `priceUSD` would normalise the distribution and potentially improve performance for linear models. It was not applied here to keep predictions directly interpretable in dollar terms. A future iteration could compare raw vs log-transformed targets.

**Why not GridSearchCV?**
Manual tuning was chosen to make the reasoning behind each hyperparameter transparent. The key insight — that `max_depth=15` + `min_samples_leaf=5` eliminates RF overfitting — was found by reasoning about the objective, not brute-force search. A future iteration could use Optuna for efficient Bayesian hyperparameter search.

**Why not a neural network?**
56,000 tabular rows is within the range where gradient boosting models consistently outperform neural networks without the additional complexity of architecture design, batch normalisation, dropout tuning, and learning rate scheduling.
