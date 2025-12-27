from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Any

import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.ar_model import AutoReg

from explainable_ai_forecast.experiments.data_preparation_preprocess import (
    PreprocSpec,
    apply_preproc,
    fit_preproc,
)

# =============================================================================
# Common output
# =============================================================================


@dataclass(frozen=True)
class BacktestResult:
    predictions: pd.DataFrame  # index=date (forecast date), cols: y_true, y_pred (+ optional cols)
    train_ends: list[pd.Timestamp]


# =============================================================================
# Snapshot helpers (for explainable AI without retraining)
# =============================================================================


def _save_oos_snapshot_ml(
    *,
    root_dir: Path,
    method: str,
    t_end: pd.Timestamp,
    model_obj: Any,
    fitted_preproc_obj: Any,
    x_fore_raw: pd.DataFrame,
    x_fore_used: pd.DataFrame,
    y_true: float,
    y_pred: float,
) -> None:
    snap_dir = root_dir / t_end.strftime("%Y-%m") / method
    snap_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_obj, snap_dir / "model.joblib")
    joblib.dump(fitted_preproc_obj, snap_dir / "preproc.joblib")  # may be None

    x_fore_raw.to_parquet(snap_dir / "X_fore_raw.parquet")
    x_fore_used.to_parquet(snap_dir / "X_fore_used.parquet")

    meta = {
        "date": str(pd.Timestamp(t_end).date()),
        "method": str(method),
        "y_true": float(y_true),
        "y_pred": float(y_pred),
    }
    pd.Series(meta).to_json(snap_dir / "meta.json")


def _save_oos_snapshot_ar(
    *,
    root_dir: Path,
    method: str,
    t_end: pd.Timestamp,
    t_fore: pd.Timestamp,
    spec: "ARSpec",
    p_used: int,
    y_tr: pd.Series,
    y_true: float,
    y_pred: float,
) -> None:
    snap_dir = root_dir / t_end.strftime("%Y-%m") / method
    snap_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "train_end": str(pd.Timestamp(t_end).date()),
        "forecast_date": str(pd.Timestamp(t_fore).date()),
        "method": str(method),
        "h": int(spec.h),
        "trend": str(spec.trend),
        "p_used": int(p_used),
        "use_bagging": bool(spec.use_bagging),
        "B_boot": int(spec.B_boot),
        "L_block": int(spec.L_block),
        "seed": int(spec.seed),
        "y_true": float(y_true),
        "y_pred": float(y_pred),
    }
    pd.Series(meta).to_json(snap_dir / "meta.json")

    y_tr.to_frame("y_train").to_parquet(snap_dir / "y_train.parquet")


# =============================================================================
# Helpers
# =============================================================================


def _drop_nan_train(X_tr: pd.DataFrame, y_tr: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    mask = X_tr.notna().all(axis=1) & y_tr.notna()
    return X_tr.loc[mask], y_tr.loc[mask]


def _months_since(anchor: pd.Timestamp, t: pd.Timestamp) -> int:
    return (t.year - anchor.year) * 12 + (t.month - anchor.month)


def _moving_block_bootstrap(arr: np.ndarray, L: int, rng: np.random.Generator) -> np.ndarray:
    """Concatène des blocs contigus de taille L tirés aléatoirement jusqu'à n."""
    n = len(arr)
    if n < 3:
        return arr.copy()
    L = max(2, min(int(L), n - 1))
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=nb)
    out = np.concatenate([arr[s : s + L] for s in starts])[:n]
    return out


# =============================================================================
# Backtest: ML models (ex: linear regression) on (X, y_future)
# =============================================================================


def pseudo_oos_expanding(
    X: pd.DataFrame,
    y_future: pd.Series,
    *,
    model,
    min_train_n: int = 36,
    start: Optional[str] = None,
    end: Optional[str] = None,
    preproc: Optional[PreprocSpec] = None,
    save_snapshots: bool = False,
    snapshots_root: Optional[Path] = None,
    method_name: str = "MODEL",
) -> BacktestResult:
    """
    Backtest pseudo out-of-sample en expanding window.

    - Train sur <= t_end
    - Predict y_future[t_end] à partir de X[t_end]
    - IMPORTANT: si preproc fourni -> fit sur train uniquement (no leakage)

    Notes:
    - on skip les dates où x_fore est incomplet ou y_true NaN
    - si save_snapshots=True: sauvegarde (model, preproc, X_fore) pour explication ex-post
    """
    if min_train_n <= 1:
        raise ValueError("min_train_n doit être > 1")

    if not X.index.equals(y_future.index):
        raise ValueError("X et y_future doivent avoir le même index")

    if save_snapshots and snapshots_root is None:
        raise ValueError("save_snapshots=True mais snapshots_root est None")

    idx = X.index
    t0 = pd.Timestamp(start) if start else idx.min()
    t1 = pd.Timestamp(end) if end else idx.max()
    valid_dates = idx[(idx >= t0) & (idx <= t1)]

    rows = []
    train_ends: list[pd.Timestamp] = []

    for t_end in valid_dates:
        X_tr = X.loc[:t_end]
        y_tr = y_future.loc[:t_end]

        X_tr, y_tr = _drop_nan_train(X_tr, y_tr)
        if len(X_tr) < min_train_n:
            continue

        x_fore = X.loc[[t_end]]
        y_true = y_future.loc[t_end]

        if x_fore.isna().any(axis=1).iloc[0] or pd.isna(y_true):
            continue

        fitted = None
        if preproc is not None:
            fitted = fit_preproc(X_tr, preproc)
            X_tr_used = apply_preproc(X_tr, fitted)
            x_fore_used = apply_preproc(x_fore, fitted, fillna_value=0.0)
        else:
            X_tr_used = X_tr
            x_fore_used = x_fore

        m = model.__class__(**model.get_params()) if hasattr(model, "get_params") else model
        m.fit(X_tr_used.to_numpy(), y_tr.to_numpy())
        y_pred = float(m.predict(x_fore_used.to_numpy())[0])

        if save_snapshots:
            _save_oos_snapshot_ml(
                root_dir=Path(snapshots_root),
                method=str(method_name),
                t_end=pd.Timestamp(t_end),
                model_obj=m,
                fitted_preproc_obj=fitted,
                x_fore_raw=x_fore,
                x_fore_used=x_fore_used,
                y_true=float(y_true),
                y_pred=float(y_pred),
            )

        rows.append({"date": t_end, "y_true": float(y_true), "y_pred": y_pred})
        train_ends.append(pd.Timestamp(t_end))

    pred_df = pd.DataFrame(rows).set_index("date").sort_index()
    return BacktestResult(predictions=pred_df, train_ends=train_ends)


# =============================================================================
# Backtest: AR(p) with periodic CV for p and optional bagging
# =============================================================================


@dataclass(frozen=True)
class ARSpec:
    h: int = 12
    min_train_n: int = 36
    trend: str = "c"

    # ✅ si défini -> p fixe (AR(p_fixed) tout le temps)
    p_fixed: Optional[int] = None

    # ✅ utilisé seulement si p_fixed is None (mode auto)
    p_grid: Iterable[int] = range(1, 13)
    cv_anchor: str = "1983-01-01"
    cv_update_every_months: int = 36

    use_bagging: bool = True
    B_boot: int = 30
    L_block: int = 12
    seed: int = 123

    # snapshots
    save_snapshots: bool = False
    snapshots_root: Optional[str] = None
    method_name: str = "AR"


def _rolling_mae_for_p(y_tr: pd.Series, p: int, h: int, min_train: int, trend: str) -> float:
    """MAE rolling à l'horizon h pour un p donné (sur y_tr, en respectant l'ordre temporel)."""
    rows = []
    last_t_end = y_tr.index.max() - relativedelta(months=h)

    for t_end in y_tr.index:
        if t_end > last_t_end:
            break

        y_sub = y_tr.loc[:t_end]
        if len(y_sub) < max(min_train, p + 1):
            continue

        m = AutoReg(y_sub, lags=p, old_names=False, trend=trend).fit()
        fc = m.predict(start=len(y_sub), end=len(y_sub) + h - 1)
        yhat_h = float(fc.iloc[-1])

        t_fore = t_end + relativedelta(months=h)
        if t_fore in y_tr.index:
            rows.append((float(y_tr.loc[t_fore]), yhat_h))

    if not rows:
        return np.inf

    y_true = np.array([r[0] for r in rows])
    y_hat = np.array([r[1] for r in rows])
    return float(mean_absolute_error(y_true, y_hat))


def _select_p_by_cv(y_tr: pd.Series, p_grid: Iterable[int], h: int, min_train: int, trend: str) -> int:
    """Sélectionne p* minimisant le MAE(h) rolling sur l'échantillon d'entraînement courant."""
    best_p, best_score = 1, np.inf
    for p in p_grid:
        score = _rolling_mae_for_p(y_tr, int(p), h, min_train, trend)
        if score < best_score:
            best_score, best_p = score, int(p)
    return int(best_p)


def _bagged_h_forecast_ARp(
    y_tr: pd.Series,
    p: int,
    h: int,
    trend: str,
    B: int,
    L: int,
    rng: np.random.Generator,
) -> float:
    """
    Prévision à horizon h via bagging (residual moving-block bootstrap) pour AR(p).
    Retourne la moyenne des prédictions bootstrapées.
    """
    base = AutoReg(y_tr, lags=p, old_names=False, trend=trend).fit()
    resid = base.resid.values
    fitted = (y_tr.iloc[-len(resid) :].values - resid)  # ŷ_t aligné

    preds = []
    for _ in range(B):
        res_b = _moving_block_bootstrap(resid, L, rng)
        y_b = fitted + res_b
        y_b = pd.Series(y_b, index=y_tr.index[-len(y_b) :])

        m_b = AutoReg(y_b, lags=p, old_names=False, trend=trend).fit()
        fc_b = m_b.predict(start=len(y_tr), end=len(y_tr) + h - 1)
        preds.append(float(fc_b.iloc[-1]))

    return float(np.mean(preds))


def backtest_ar_expanding(y: pd.Series, spec: ARSpec) -> BacktestResult:
    """
    Backtest pseudo-OOS expanding pour AR(p), avec:
    - mode auto: sélection périodique de p via CV rolling MAE(h) sur le train
    - mode fixe: p_fixed impose AR(p_fixed) partout
    - bagging optionnel (moving-block bootstrap des résidus)

    Sortie:
    - predictions indexées par la date forecastée (t_end + h)
    - train_ends = dates t_end (fin du train) utilisées

    Si spec.save_snapshots=True:
    - sauvegarde y_train + meta (p_used, etc.) pour explication ex-post
    """
    y = y.astype(float).copy()
    rng = np.random.default_rng(spec.seed)

    if spec.h <= 0:
        raise ValueError("spec.h doit être > 0")
    if spec.min_train_n <= 1:
        raise ValueError("spec.min_train_n doit être > 1")
    if spec.p_fixed is not None and spec.p_fixed <= 0:
        raise ValueError("spec.p_fixed doit être > 0")

    if spec.save_snapshots and spec.snapshots_root is None:
        raise ValueError("spec.save_snapshots=True mais spec.snapshots_root est None")

    cv_anchor = pd.Timestamp(spec.cv_anchor)
    last_t_end = y.index.max() - relativedelta(months=spec.h)

    rows = []
    train_ends: list[pd.Timestamp] = []
    current_p: Optional[int] = None  # uniquement utilisé en mode auto

    for t_end in y.index:
        if t_end > last_t_end:
            break

        y_tr = y.loc[:t_end]
        if len(y_tr) < spec.min_train_n:
            continue

        # --- choisir p_used (fixe ou auto) ---
        if spec.p_fixed is not None:
            p_used = int(spec.p_fixed)
        else:
            need_cv = False
            if t_end >= cv_anchor:
                m = _months_since(cv_anchor, t_end)
                need_cv = (m % spec.cv_update_every_months == 0)

            if current_p is None and not need_cv:
                current_p = 1
            if need_cv:
                current_p = _select_p_by_cv(y_tr, spec.p_grid, spec.h, spec.min_train_n, spec.trend)

            p_used = int(current_p)

        t_fore = t_end + relativedelta(months=spec.h)
        if t_fore not in y.index:
            continue

        y_true = y.loc[t_fore]
        if pd.isna(y_true):
            continue

        # forecast
        if spec.use_bagging:
            y_pred = _bagged_h_forecast_ARp(
                y_tr=y_tr,
                p=p_used,
                h=spec.h,
                trend=spec.trend,
                B=spec.B_boot,
                L=spec.L_block,
                rng=rng,
            )
        else:
            m_ar = AutoReg(y_tr, lags=p_used, old_names=False, trend=spec.trend).fit()
            fc = m_ar.predict(start=len(y_tr), end=len(y_tr) + spec.h - 1)
            y_pred = float(fc.iloc[-1])

        if spec.save_snapshots:
            _save_oos_snapshot_ar(
                root_dir=Path(spec.snapshots_root),
                method=str(spec.method_name),
                t_end=pd.Timestamp(t_end),
                t_fore=pd.Timestamp(t_fore),
                spec=spec,
                p_used=int(p_used),
                y_tr=y_tr,
                y_true=float(y_true),
                y_pred=float(y_pred),
            )

        rows.append({"date": t_fore, "y_true": float(y_true), "y_pred": float(y_pred), "p_used": p_used})
        train_ends.append(pd.Timestamp(t_end))

    pred_df = pd.DataFrame(rows).set_index("date").sort_index()
    return BacktestResult(predictions=pred_df, train_ends=train_ends)