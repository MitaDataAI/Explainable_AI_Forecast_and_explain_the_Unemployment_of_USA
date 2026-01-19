from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("default")


def _snapshot_dir_from_date(snapshots_root: Path, dt: pd.Timestamp, method: str) -> Path:
    """
    Repo snapshot layout:
      <snapshots_root>/<YYYY-MM>/<METHOD>/
    """
    ym = pd.Timestamp(dt).strftime("%Y-%m")
    return snapshots_root / ym / method


def main():
    p = argparse.ArgumentParser(description="Plot SHAP functional forms (OOS) with snapshot-centering")
    p.add_argument("--data-path", required=True, help="Path to original feature dataset (CSV)")
    p.add_argument("--shap-path", required=True, help="Path to SHAP parquet file (already computed)")
    p.add_argument("--snapshots-root", required=True, help="Root dir of OOS snapshots (contains YYYY-MM/METHOD)")
    p.add_argument("--method", default="LINREG", help="Snapshot method folder name (e.g., LINREG)")
    p.add_argument("--date-col", default="date", help="Name of date column")
    p.add_argument("--out-dir", required=True, help="Output directory for figures")
    p.add_argument("--features", nargs="+", required=True, help="List of features to plot")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots_root = Path(args.snapshots_root)
    method = str(args.method)

    # --- Load SHAP (already computed)
    shap_df = pd.read_parquet(args.shap_path)
    shap_df[args.date_col] = pd.to_datetime(shap_df[args.date_col])
    shap_df = shap_df.sort_values(args.date_col).reset_index(drop=True)

    # --- Load original X data
    X = pd.read_csv(args.data_path, parse_dates=[args.date_col]).set_index(args.date_col)

    # --- Strict OOS alignment (critical)
    dates = pd.to_datetime(shap_df[args.date_col])
    X_oos = X.loc[dates]

    # --- Metadata columns (not SHAP)
    meta_cols = ["date", "method", "y_true", "y_pred"]
    shap_features = [c for c in shap_df.columns if c not in meta_cols]

    # Cache per month to avoid re-loading train file 729 times
    mu_cache: dict[tuple[str, str], pd.Series] = {}  # key=(YYYY-MM, method) -> mean series over train_x

    for feat in args.features:
        if feat not in shap_features:
            print(f"⚠️ Feature '{feat}' not found in SHAP file → skipped")
            continue
        if feat not in X.columns:
            print(f"⚠️ Feature '{feat}' not found in data file → skipped")
            continue

        x_centered = []
        shap_vals = shap_df[feat].to_numpy()

        for i, dt in enumerate(dates):
            ym = dt.strftime("%Y-%m")
            key = (ym, method)

            if key not in mu_cache:
                snap_dir = _snapshot_dir_from_date(snapshots_root, dt, method)
                train_path = snap_dir / "X_train_used.parquet"
                if not train_path.exists():
                    raise FileNotFoundError(f"Missing baseline file: {train_path}")
                train_x = pd.read_parquet(train_path)

                # mean over train window (baseline μ_{t})
                mu_cache[key] = train_x.mean(numeric_only=True)

            mu_t = mu_cache[key]
            # Center by snapshot baseline: x* = x - μ_{t}
            x_centered.append(float(X_oos.iloc[i][feat] - mu_t.get(feat, 0.0)))

        # --- Plot: SHAP vs centered x (matches author)
        plt.figure(figsize=(6, 4))
        plt.scatter(
            x_centered,
            shap_vals,
            alpha=0.3,
            s=15
        )
        plt.axhline(0, color="black", lw=1)
        plt.axvline(0, color="black", lw=1)
        plt.xlabel(f"{feat} (centered: x - mean_train_snapshot)")
        plt.ylabel(f"SHAP({feat})")
        plt.title(f"{method} – Functional form (rolling OOS, centered)")
        plt.tight_layout()

        out_path = out_dir / f"shap_functional_centered_{feat}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()