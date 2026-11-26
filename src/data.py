"""Load, clean and split the AgriPak dataset.

Run:  python src/data.py
Outputs (data/processed/): crop_data_clean.csv, train.csv, test.csv
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (BINARY, CATEGORICAL, CLEAN_PATH, NUMERIC, PROCESSED_DIR,
                    RANDOM_SEED, RAW_PATH, TARGET, TEST_SIZE, TEST_PATH,
                    TRAIN_PATH)


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every cleaning decision. Each step is documented in the README."""
    df = df.copy()

    # 1. Normalise column names and text values (strip / consistent case).
    df.columns = [c.strip().lower() for c in df.columns]
    for col in CATEGORICAL:
        df[col] = df[col].astype("string").str.strip().str.title()

    # 2. Fix data types.
    for col in NUMERIC + BINARY + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Drop rows with a missing target (unusable for supervised learning).
    df = df.dropna(subset=[TARGET])

    # 4. Impute remaining missing values (none in the shipped file, but the
    #    script must stay robust): median for numeric, mode for categorical/binary.
    for col in NUMERIC:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL + BINARY:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    # 5. Remove exact duplicate rows.
    df = df.drop_duplicates()

    # 6. Final dtypes: ints for flags/target, plain strings for text columns.
    for col in BINARY + [TARGET, "days_to_harvest"]:
        df[col] = df[col].round().astype(int)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)

    return df.reset_index(drop=True)


def split(df: pd.DataFrame):
    """Stratified 80/20 split with a fixed seed (reproducible)."""
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED,
                            stratify=df[TARGET])


def main():
    raw = load_raw()
    df = clean(raw)
    print(f"Raw rows: {len(raw)} | after cleaning: {len(df)} "
          f"| missing values left: {int(df.isna().sum().sum())}")

    train, test = split(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"Train: {len(train)} rows (positive rate {train[TARGET].mean():.3f})")
    print(f"Test : {len(test)} rows (positive rate {test[TARGET].mean():.3f})")


if __name__ == "__main__":
    main()
