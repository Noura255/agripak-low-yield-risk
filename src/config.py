"""Shared constants so every script agrees on paths, seed and column names."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "crop_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_PATH = PROCESSED_DIR / "crop_data_clean.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
REPORTS_DIR = ROOT / "reports"

TARGET = "low_yield_risk"
RANDOM_SEED = 42
TEST_SIZE = 0.20
LEAKAGE_THRESHOLD = 0.90

CATEGORICAL = ["region", "soil_type", "crop", "weather_condition"]
NUMERIC = ["rainfall_mm", "temperature_c", "days_to_harvest"]
BINARY = ["fertilizer_used", "irrigation_used"]
