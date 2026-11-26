import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from config import TARGET
from data import clean, load_raw, split


def test_no_missing_after_cleaning():
    assert clean(load_raw()).isna().sum().sum() == 0


def test_split_is_reproducible():
    df = clean(load_raw())
    a_train, a_test = split(df)
    b_train, b_test = split(df)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)


def test_split_is_stratified_and_80_20():
    df = clean(load_raw())
    train, test = split(df)
    assert abs(len(test) / len(df) - 0.20) < 0.001
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.001


def test_no_row_overlap():
    train, test = split(clean(load_raw()))
    assert train.index.intersection(test.index).empty
