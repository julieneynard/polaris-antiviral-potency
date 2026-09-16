"""official_chronological_split must produce a disjoint, complete, order-preserving
partition purely from the dataset's own `Set` column - no random splitting anywhere.

No network access: operates on a synthetic in-memory DataFrame, never calls
po.load_dataset.
"""

from __future__ import annotations

import pandas as pd

from src.data import official_chronological_split


def _make_df(splits: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "molecule_name": [f"mol_{i}" for i in range(len(splits))],
            "cxsmiles": ["C"] * len(splits),
            "split": splits,
            "target": list(range(len(splits))),
        }
    )


def test_split_is_disjoint_and_complete():
    df = _make_df(["Train", "Test", "Train", "Train", "Test"])

    result = official_chronological_split(df)

    train_names = set(result.train["molecule_name"])
    test_names = set(result.test["molecule_name"])

    assert train_names.isdisjoint(test_names)
    assert train_names | test_names == set(df["molecule_name"])
    assert len(result.train) == 3
    assert len(result.test) == 2


def test_split_only_uses_the_set_column_not_row_order_or_content():
    # Shuffle the rows and use non-sequential target values; the split must still be
    # determined purely by the "split" column value of each row, not position or index.
    df = _make_df(["Test", "Train", "Test", "Train"])
    df["target"] = [100, -5, 42, 0]

    result = official_chronological_split(df)

    assert set(result.train["target"]) == {-5, 0}
    assert set(result.test["target"]) == {100, 42}


def test_split_preserves_all_columns():
    df = _make_df(["Train", "Test"])

    result = official_chronological_split(df)

    assert list(result.train.columns) == list(df.columns)
    assert list(result.test.columns) == list(df.columns)
