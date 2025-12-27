from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Optional, Tuple, List, Dict
from math import sqrt, erf, isfinite

# ============================================================
# 0) Utils
# ============================================================

def _ensure_wide(df_wide: pd.DataFrame) -> pd.DataFrame:
    wide = df_wide.copy()
    if "date" in wide.columns:
        wide["date"] = pd.to_datetime(wide["date"], errors="coerce")
        wide = wide.set_index("date")
    if not isinstance(wide.index, pd.DatetimeIndex):
        raise ValueError("df_wide must have DatetimeIndex or a 'date' column.")
    wide = wide.sort_index()
    if "true" not in wide.columns:
        raise ValueError("df_wide must contain column 'true'.")
    return wide


def _resolve_methods(wide: pd.DataFrame, methods: Optional[Iterable[str]]) -> List[str]:
    if methods is None:
        return [c for c in wide.columns if c != "true"]
    return [m for m in methods if m in wide.columns and m != "true"]


def _build_windows(
    wide: pd.DataFrame,
    periods: List[Tuple[str, Optional[str], str]],
    *,
    include_overall: bool = True,
    overall_label: str = "Ensemble",
):
    full_start, full_end = wide.index.min(), wide.index.max()
    windows = []
    if include_overall:
        windows.append((full_start, full_end, overall_label))
    for start, end, label in periods:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end) if end is not None else full_end
        windows.append((s, e, label))
    return windows


def _mae_and_errors_for_window(sub, methods, *, min_obs):
    maes, err_abs = {}, {}
    for m in methods:
        diffs = (sub["true"] - sub[m]).abs().dropna()
        err_abs[m] = diffs
        maes[m] = diffs.mean() if len(diffs) >= min_obs else np.nan
    return maes, err_abs


def _phi(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _dm_pvalue(diff: np.ndarray, lags: int):
    x = diff[np.isfinite(diff)]
    T = len(x)
    if T < 3:
        return np.nan
    dbar = x.mean()
    gamma0 = np.mean((x - dbar) ** 2)
    var = gamma0
    for k in range(1, min(lags, T - 1) + 1):
        w = 1.0 - k / (lags + 1)
        cov = np.mean((x[k:] - dbar) * (x[:-k] - dbar))
        var += 2 * w * cov
    if var <= 0:
        return np.nan
    stat = dbar / np.sqrt(var / T)
    return 2 * (1 - _phi(abs(stat)))


def make_mae_dm_pivot(
    df_wide: pd.DataFrame,
    periods: List[Tuple[str, Optional[str], str]],
    *,
    methods: Optional[Iterable[str]] = None,
    include_overall: bool = True,
    overall_label: str = "Ensemble",
    min_obs: int = 20,
    round_digits: int = 4,
    dm_lags: int = 0,
):
    wide = _ensure_wide(df_wide)
    methods = _resolve_methods(wide, methods)
    windows = _build_windows(wide, periods, include_overall=include_overall, overall_label=overall_label)

    rows = []
    for start, end, label in windows:
        sub = wide.loc[start:end, ["true"] + methods].dropna(subset=["true"])
        maes, err_abs = _mae_and_errors_for_window(sub, methods, min_obs=min_obs)
        best = min((m for m in methods if isfinite(maes[m])), key=lambda m: maes[m], default=None)

        for m in methods:
            if not isfinite(maes[m]):
                rows.append((m, label, np.nan))
                continue
            cell = f"{maes[m]:.{round_digits}f}"
            if best and m != best:
                common = err_abs[m].index.intersection(err_abs[best].index)
                if len(common) >= min_obs:
                    diff = err_abs[m].loc[common].values - err_abs[best].loc[common].values
                    p = _dm_pvalue(diff, dm_lags)
                    if isfinite(p):
                        cell += f" ({p:.3f})"
            rows.append((m, label, cell))

    df = pd.DataFrame(rows, columns=["model", "period", "MAE"])
    return df.pivot(index="model", columns="period", values="MAE")