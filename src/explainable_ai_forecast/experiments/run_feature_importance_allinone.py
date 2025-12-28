from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--compare-dir", type=str, required=True)
    p.add_argument("--snapshots-subdir", type=str, default="oos_snapshots")

    # Méthode à expliquer (par défaut LINREG car c’est celui qui a model.joblib + X_fore_used)
    p.add_argument("--method", type=str, default="LINREG")

    # Sampling/Perf
    p.add_argument("--max-rows", type=int, default=0, help="0 = toutes les dates, sinon sous-échantillonne")
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--out-name", type=str, default="xai_perm_shap_all.csv")

    # NEW: export panel SHAP/contributions par snapshot
    p.add_argument(
        "--out-shap-snapshots",
        type=str,
        default="",
        help="Si non vide, exporte un CSV long: date, model, feature, shap_value, y_true (ex: shap_snapshots_long.csv)",
    )
    p.add_argument(
        "--date-key",
        type=str,
        default="date",
        help="Clé dans meta.json contenant la date du snapshot si disponible (sinon fallback sur le dossier YYYY-MM).",
    )

    return p.parse_args()


def _is_linear_sklearn(model: Any) -> bool:
    return hasattr(model, "coef_") and hasattr(model, "intercept_") and hasattr(model, "predict")


def _predict_one(model: Any, x_row: np.ndarray) -> float:
    # x_row shape: (p,)
    y = model.predict(x_row.reshape(1, -1))
    return float(np.asarray(y).reshape(-1)[0])


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = (y_true - y_pred)
    return float(np.mean(e * e))


def _collect_snapshots(snapshots_root: Path, method: str) -> list[Path]:
    # snapshots_root/YYYY-MM/METHOD/
    out = []
    if not snapshots_root.exists():
        raise FileNotFoundError(f"Snapshots introuvables: {snapshots_root}")

    for ym_dir in sorted([p for p in snapshots_root.iterdir() if p.is_dir()]):
        mdir = ym_dir / method
        if mdir.exists() and mdir.is_dir():
            out.append(mdir)
    if not out:
        raise FileNotFoundError(f"Aucun snapshot trouvé pour method='{method}' dans {snapshots_root}")
    return out


def _read_snapshot(mdir: Path) -> tuple[pd.DataFrame, float, Any, dict]:
    # Files: model.joblib, X_fore_used.parquet, meta.json
    model_path = mdir / "model.joblib"
    x_path = mdir / "X_fore_used.parquet"
    meta_path = mdir / "meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing: {model_path}")
    if not x_path.exists():
        raise FileNotFoundError(f"Missing: {x_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")

    model = joblib.load(model_path)
    X = pd.read_parquet(x_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    y_true = float(meta["y_true"])
    return X, y_true, model, meta


def _align_X(X_list: list[pd.DataFrame]) -> pd.DataFrame:
    # Union columns, fill missing with 0.0
    cols = sorted(set().union(*[set(x.columns) for x in X_list]))
    aligned = []
    for x in X_list:
        xx = x.reindex(columns=cols)
        aligned.append(xx)
    X_all = pd.concat(aligned, axis=0, ignore_index=True)
    return X_all.astype(float).fillna(0.0)


def _baseline_predictions(models: list[Any], X: np.ndarray) -> np.ndarray:
    yhat = np.empty(X.shape[0], dtype=float)
    for i, (m, xi) in enumerate(zip(models, X)):
        yhat[i] = _predict_one(m, xi)
    return yhat


def _permute_importance(
    models: list[Any],
    X: np.ndarray,
    y_true: np.ndarray,
    seed: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Permutation importance "cross-time" :
    - On permute une colonne de X entre les snapshots
    - On prédit avec chaque modèle_i sur son x_i permuté
    - Importance = (loss_perm - loss_base)
      - abs error : MAE
      - deviance : MSE
    """
    rng = np.random.default_rng(seed)

    base_pred = _baseline_predictions(models, X)
    base_mae = _mae(y_true, base_pred)
    base_mse = _mse(y_true, base_pred)

    n, p = X.shape
    d_mae = np.zeros(p, dtype=float)
    d_mse = np.zeros(p, dtype=float)

    for j in range(p):
        perm_idx = rng.permutation(n)
        Xp = X.copy()
        Xp[:, j] = Xp[perm_idx, j]

        pred_p = np.empty(n, dtype=float)
        for i, (m, xi) in enumerate(zip(models, Xp)):
            pred_p[i] = _predict_one(m, xi)

        d_mae[j] = _mae(y_true, pred_p) - base_mae
        d_mse[j] = _mse(y_true, pred_p) - base_mse

    return pd.Series(d_mae), pd.Series(d_mse)


def _shapley_share_linear(
    models: list[Any],
    X_df: pd.DataFrame,
) -> pd.Series:
    """
    Shapley share (linéaire, stable, sans dépendre de shap):
    Pour chaque snapshot i:
      contrib_j = coef_{i,j} * (x_{i,j} - mu_j)
    share_j = mean_i( |contrib_j| / sum_k |contrib_k| )
    """
    X = X_df.to_numpy()
    mu = np.nanmean(X, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    n, p = X.shape
    shares = np.zeros(p, dtype=float)

    for i, m in enumerate(models):
        if not _is_linear_sklearn(m):
            # si un modèle n’est pas linéaire -> on skip ce snapshot
            continue
        coef = np.asarray(m.coef_).reshape(-1)
        if coef.shape[0] != p:
            # colonnes incompatibles -> skip
            continue

        contrib = coef * (X[i] - mu)
        denom = np.sum(np.abs(contrib))
        if denom == 0 or not np.isfinite(denom):
            continue
        shares += np.abs(contrib) / denom

    # moyenne sur n (même si certains snapshots ont été skip, on normalise par n pour rester conservateur)
    shares = shares / max(1, n)
    return pd.Series(shares, index=X_df.columns)


def _snapshot_date(meta: dict, mdir: Path, date_key: str) -> pd.Timestamp:
    """
    1) meta[date_key] si présent,
    2) fallback: dossier YYYY-MM -> YYYY-MM-01
       mdir = .../YYYY-MM/METHOD
    """
    if isinstance(meta, dict) and date_key in meta:
        try:
            return pd.to_datetime(meta[date_key])
        except Exception:
            pass

    ym = mdir.parent.name  # "YYYY-MM"
    try:
        return pd.to_datetime(f"{ym}-01")
    except Exception:
        return pd.NaT


def _append_shap_snapshot_rows(
    out_csv: Path,
    *,
    date_t: pd.Timestamp,
    model: str,
    feature_names: list[str],
    shap_values: np.ndarray,
    y_true: float,
) -> None:
    """
    Append CSV: date | model | feature | shap_value | y_true
    shap_values doit être (p,)
    """
    sv = np.asarray(shap_values).reshape(-1)
    if sv.shape[0] != len(feature_names):
        raise ValueError("Mismatch shap_values / feature_names")

    df_out = pd.DataFrame(
        {
            "date": [pd.to_datetime(date_t)] * len(feature_names),
            "model": [model] * len(feature_names),
            "feature": feature_names,
            "shap_value": sv.astype(float),
            "y_true": [float(y_true)] * len(feature_names),
        }
    )

    header = not out_csv.exists()
    df_out.to_csv(out_csv, mode="a", header=header, index=False)


def main() -> None:
    args = parse_args()

    compare_dir = Path(args.compare_dir)
    snapshots_root = compare_dir / args.snapshots_subdir
    method = args.method

    snap_dirs = _collect_snapshots(snapshots_root, method)

    # Read all snapshots
    X_list: list[pd.DataFrame] = []
    y_list: list[float] = []
    models: list[Any] = []
    metas: list[dict] = []
    snap_dirs_kept: list[Path] = []

    for mdir in snap_dirs:
        X_i, y_true_i, model_i, meta_i = _read_snapshot(mdir)
        X_list.append(X_i)
        y_list.append(y_true_i)
        models.append(model_i)
        metas.append(meta_i)
        snap_dirs_kept.append(mdir)

    # Optional sampling
    rng = np.random.default_rng(args.seed)
    n_total = len(models)
    if args.max_rows and args.max_rows > 0 and args.max_rows < n_total:
        idx = rng.choice(n_total, size=int(args.max_rows), replace=False)
        idx = np.sort(idx)
        X_list = [X_list[i] for i in idx]
        y_list = [y_list[i] for i in idx]
        models = [models[i] for i in idx]
        metas = [metas[i] for i in idx]
        snap_dirs_kept = [snap_dirs_kept[i] for i in idx]

    # Align X (union des colonnes)
    X_all_df = _align_X(X_list)
    y_true = np.asarray(y_list, dtype=float)
    X_all = X_all_df.to_numpy()

    # Calcul mu une fois (utilisé pour contributions linéaires par snapshot)
    mu = np.nanmean(X_all, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # NEW: Export shap snapshots (contributions linéaires) si demandé
    if args.out_shap_snapshots:
        out_snap_path = compare_dir / args.out_shap_snapshots
        if out_snap_path.exists():
            out_snap_path.unlink()  # évite d'append plusieurs runs par erreur

        cols = list(X_all_df.columns)
        for mdir, Xi, yi, mi, meta_i in zip(snap_dirs_kept, X_list, y_list, models, metas):
            if not _is_linear_sklearn(mi):
                continue

            dt = _snapshot_date(meta_i, mdir, args.date_key)
            if pd.isna(dt):
                continue

            # aligne la ligne snapshot sur cols
            x_row = Xi.reindex(columns=cols).astype(float).fillna(0.0).to_numpy().reshape(-1)

            coef = np.asarray(mi.coef_).reshape(-1)
            if coef.shape[0] != len(cols):
                continue

            # contribution linéaire par feature
            contrib = coef * (x_row - mu)

            _append_shap_snapshot_rows(
                out_csv=out_snap_path,
                date_t=dt,
                model=method,
                feature_names=cols,
                shap_values=contrib,
                y_true=yi,
            )

        print(f"✅ SHAP snapshots exportés: {out_snap_path}")

    # 1) Permutation importances
    d_mae, d_mse = _permute_importance(models=models, X=X_all, y_true=y_true, seed=args.seed)

    # 2) Shapley share (linéaire)
    shap_share = _shapley_share_linear(models=models, X_df=X_all_df)

    # Build tidy output
    out = pd.DataFrame(
        {
            "model": method,
            "feature": list(X_all_df.columns),
            "mean_perm_abs_error": d_mae.to_numpy(),
            "mean_perm_deviance": d_mse.to_numpy(),
            "shapley_share": shap_share.reindex(X_all_df.columns).to_numpy(),
            "n_snapshots": int(len(models)),
        }
    )

    out_path = compare_dir / args.out_name
    out.to_csv(out_path, index=False)

    print("✅ XAI terminé")
    print(f"✅ rows (features): {len(out)}")
    print(f"✅ snapshots utilisés: {len(models)} / {n_total}")
    print(f"✅ fichier: {out_path}")
    print("\nAperçu:")
    print(out.sort_values("mean_perm_abs_error", ascending=False).head(15))


if __name__ == "__main__":
    main()