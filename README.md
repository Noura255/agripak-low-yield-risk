# AgriPak Farms — Low-Yield Risk Prediction

Machine Learning Internship, 7-week client project. AgriPak Farms wants a model that flags
fields at risk of a low yield **before harvest** so they can act in time.
This repository is built up week by week; this is the **Week 1** state
(data ingestion, cleaning, reproducible split, baseline, leakage check).

## Repository structure

```
README.md
LICENSE
.gitignore
requirements.txt
run_pipeline.py        # runs the whole Week-1 pipeline
data/raw/              # crop_data.csv (as shared by the client, never edited)
data/processed/        # generated: clean data + train/test split (git-ignored)
src/                   # config.py, data.py, baseline.py, leakage_check.py
notebooks/             # exploration (empty for now)
reports/               # generated: baseline_scores.json, feature_target_correlation.csv
tests/                 # pytest checks (cleaning, split reproducibility)
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
python run_pipeline.py        # clean -> split -> baseline -> leakage scan
pytest                        # optional: run the tests
```

Individual steps: `python src/data.py`, `python src/baseline.py`, `python src/leakage_check.py`.
The split uses `random_state=42` and `stratify=low_yield_risk`, so every run produces the same files.

## Dataset

50,000 rows, 10 columns. Target: `low_yield_risk` (1 = at risk, 0 = not), 35% positive.

| Type | Columns |
|---|---|
| Categorical | `region`, `soil_type`, `crop`, `weather_condition` |
| Numeric | `rainfall_mm`, `temperature_c`, `days_to_harvest` |
| Binary | `fertilizer_used`, `irrigation_used` |

## Cleaning decisions (`src/data.py`)

| # | Decision | Reason / outcome on this file |
|---|---|---|
| 1 | Lower-case and strip column names; strip whitespace and title-case text values | Prevents `"clay"` / `"Clay "` becoming separate categories. No changes were needed. |
| 2 | Cast numeric, binary and target columns with `pd.to_numeric(errors="coerce")` | Guarantees correct dtypes; bad values become NaN and are handled by 3–4. No bad values found. |
| 3 | Drop rows with a missing target | A row without a label cannot be used for supervised learning. 0 rows dropped. |
| 4 | Impute remaining missing values: median for numeric, mode for categorical/binary | Robust to outliers and keeps all rows. **0 missing values** in the shipped file, so nothing was imputed; the code stays in place for future data drops. |
| 5 | Drop exact duplicate rows | 0 duplicates found. |
| 6 | Final dtypes: `int` for binary flags, target and `days_to_harvest`; `str` for categoricals | Consistent types for modelling. |
| 7 | No outlier removal | Value ranges are plausible (rainfall 100–1000 mm, temperature 15–40 °C, harvest 60–149 days) and no value is physically impossible. |

Result: 50,000 rows in, 50,000 rows out.

## Train/test split

Stratified 80/20 (`random_state=42`): 40,000 train / 10,000 test rows, positive rate 0.350 in both.
Scaling and one-hot encoding live inside a scikit-learn `Pipeline`, so they are fitted on the training set only.

## Baseline: Logistic Regression (test set)

| Metric | Score |
|---|---|
| Accuracy | 0.9187 |
| Precision | 0.8886 |
| Recall | 0.8777 |
| F1 | 0.8831 |

For reference, always predicting "not at risk" gives 0.65 accuracy, so the model is clearly learning signal.
Because the classes are 65/35, F1 / recall matter more than accuracy for the "flag at-risk fields" goal.

## Leakage sanity check (`src/leakage_check.py`)

Pearson correlation of every feature with the target, computed on the training set only
(categorical columns one-hot encoded, so each level has its own value). Threshold: |r| > 0.9.

**Flagged columns: none.** The highest correlations were:

| Feature | r with target | Verdict |
|---|---|---|
| `rainfall_mm` | -0.637 | Genuine predictor. Below the threshold, and agronomically sensible: low rainfall → higher risk of low yield. Measured before harvest. |
| `fertilizer_used` | -0.336 | Genuine predictor. A management decision taken during the season. |
| `irrigation_used` | -0.265 | Genuine predictor. Same reasoning; irrigation offsets low rainfall. |
| `temperature_c` | -0.067 | Weak genuine predictor. |
| all other columns | \|r\| < 0.011 | Essentially uncorrelated (including `days_to_harvest`, `region`, `soil_type`, `crop`, `weather_condition`). |

Notes and caveats:
- No column crosses 0.9, so nothing is dropped. The 0.919 baseline accuracy is explained by `rainfall_mm`, `fertilizer_used` and `irrigation_used`, none of which is a post-harvest quantity.
- Correlation only catches strong *linear* leakage. It will not detect leakage that is non-linear, interaction-based, or hidden in how the label was created. Later weeks should add permutation importance and a check on how `low_yield_risk` was derived.
- `days_to_harvest` is only legitimate if it is known at prediction time (e.g. planned harvest date). To confirm with the client.
- The data looks very clean and synthetic (no missing values, uniform ranges), so real-world performance may be lower.

## Roadmap

Later weeks build on this repository (stronger models, validation, tuning, explainability, deployment).
