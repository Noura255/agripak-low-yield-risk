"""Leakage sanity check: correlation of every feature with the target.

Categorical columns are one-hot encoded so each level gets its own correlation.
Computed on the TRAINING set only. Run: python src/leakage_check.py
"""
import pandas as pd

from config import (CATEGORICAL, LEAKAGE_THRESHOLD, REPORTS_DIR, TARGET,
                    TRAIN_PATH)


def feature_target_correlation(df: pd.DataFrame) -> pd.Series:
    X = pd.get_dummies(df, columns=CATEGORICAL, dtype=int)
    corr = X.corr()[TARGET].drop(TARGET)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def main():
    corr = feature_target_correlation(pd.read_csv(TRAIN_PATH))
    REPORTS_DIR.mkdir(exist_ok=True)
    corr.rename("correlation_with_target").to_csv(REPORTS_DIR / "feature_target_correlation.csv")

    print(corr.round(4).to_string())
    flagged = corr[corr.abs() > LEAKAGE_THRESHOLD]
    print(f"\nFlagged (|r| > {LEAKAGE_THRESHOLD}): "
          f"{list(flagged.index) if len(flagged) else 'none'}")
    print(f"Strongest feature: {corr.index[0]} (r = {corr.iloc[0]:.3f})")


if __name__ == "__main__":
    main()
