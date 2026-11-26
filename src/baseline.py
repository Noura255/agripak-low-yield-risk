"""Logistic Regression baseline.  Run: python src/baseline.py"""
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (BINARY, CATEGORICAL, NUMERIC, RANDOM_SEED, REPORTS_DIR,
                    TARGET, TEST_PATH, TRAIN_PATH)


def build_model() -> Pipeline:
    # Scaler / encoder live inside the Pipeline so they are fitted on train only.
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))])


def main():
    train, test = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    features = NUMERIC + CATEGORICAL + BINARY
    model = build_model().fit(train[features], train[TARGET])
    pred = model.predict(test[features])

    scores = {
        "accuracy": accuracy_score(test[TARGET], pred),
        "precision": precision_score(test[TARGET], pred),
        "recall": recall_score(test[TARGET], pred),
        "f1": f1_score(test[TARGET], pred),
        "majority_class_accuracy": 1 - test[TARGET].mean(),
    }
    scores = {k: round(float(v), 4) for k, v in scores.items()}
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "baseline_scores.json").write_text(json.dumps(scores, indent=2))
    for k, v in scores.items():
        print(f"{k:>24}: {v:.4f}")


if __name__ == "__main__":
    main()
