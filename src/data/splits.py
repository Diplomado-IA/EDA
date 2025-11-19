"""Splits cronológicos y generadores de CV temporal."""
from typing import Tuple, Optional, Any, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, KFold


def _ensure_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series)
    except Exception:
        return series


def chronological_split(df: pd.DataFrame, date_col: str, train_end: Any, test_start: Any, target: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    assert date_col in df.columns, f"{date_col} no está en df"
    assert target in df.columns, f"{target} no está en df"
    dc = df[date_col]
    if np.issubdtype(dc.dtype, np.number):
        train_mask = dc <= train_end
        test_mask = dc >= test_start
    else:
        dc = _ensure_datetime(dc)
        train_end_dt = pd.to_datetime(train_end)
        test_start_dt = pd.to_datetime(test_start)
        train_mask = dc <= train_end_dt
        test_mask = dc >= test_start_dt
    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]
    return X_train, y_train, X_test, y_test


def get_time_series_cv(n_splits: int = 5, gap: int = 0, max_train_size: Optional[int] = None) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits, gap=gap, max_train_size=max_train_size)


def get_cv(cfg: Dict[str, Any]):
    kind = cfg.get("kind", "kfold")
    if kind == "time":
        return get_time_series_cv(cfg.get("n_splits", 5), cfg.get("gap", 0), cfg.get("max_train_size", None))
    return KFold(n_splits=cfg.get("n_splits", 5), shuffle=cfg.get("shuffle", False), random_state=cfg.get("random_state", 42))
