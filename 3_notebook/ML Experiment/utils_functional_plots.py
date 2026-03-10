import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0d0d0d",
    "axes.facecolor":   "#111111",
    "savefig.facecolor":"#0d0d0d",
    "text.color":       "#e5e5e5",
    "axes.edgecolor":   "#bfbfbf",
    "axes.labelcolor":  "#e5e5e5",
    "xtick.color":      "#d8d8d8",
    "ytick.color":      "#d8d8d8",
    "grid.color":       "#5a5a5a",
})

_DARK_LINE = "#ffffff"
_DARK_ZERO = "#7a5a5a"
_DARK_NA   = "#a5a5a5"
_DARK_GRID = (0.25, ":")

_DARK_COLORS = (
    "#7fdfff",
    "#ffd166",
    "#ff7f7f",
    "#9d7aff",
    "#8bd17c",
    "#f29ae1",
)


def build_functional_inputs_from_meta(
    *,
    meta_models: dict,
    ts_data: pd.DataFrame,
    end_date=None,
    date_col: str = "ds",
    model_order=("LR", "RIDGE", "LGBM"),
):
    """
    Construit les objets nécessaires aux functional plots à partir de
    meta_models['bundles'] et des données temporelles.

    Retour
    ------
    models_dict : dict
        models_dict[model_name] = (model_selected, prep_selected)

    X_dict : dict
        X_dict[model_name] = DataFrame des features disponibles

    features_dict : dict
        features_dict[model_name] = liste des features effectivement présentes
    """
    if "bundles" not in meta_models or not meta_models["bundles"]:
        raise ValueError("meta_models['bundles'] est vide ou absent.")

    ts_base = ts_data.copy()
    ts_base[date_col] = (
        pd.to_datetime(ts_base[date_col], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp(how="start")
        .dt.normalize()
    )
    ts_base = ts_base.dropna(subset=[date_col]).reset_index(drop=True)

    if end_date is not None:
        end_date = (
            pd.Timestamp(end_date)
            .to_period("M")
            .to_timestamp(how="start")
            .normalize()
        )

    models_dict = {}
    X_dict = {}
    features_dict = {}

    for model_name in model_order:
        bundle = meta_models["bundles"].get(model_name, None)
        if bundle is None:
            print(f"⚠️ Bundle absent pour {model_name}")
            continue

        models_all = list(bundle.get("models", []))
        dates_all = pd.to_datetime(bundle.get("train_fit_dates", []), errors="coerce")
        feats_all = list(bundle.get("features", [])) if bundle.get("features", None) is not None else []
        prep_all = bundle.get("preprocs", None)

        if len(models_all) == 0:
            print(f"⚠️ Aucun modèle fitted pour {model_name}")
            continue

        if len(feats_all) == 0:
            print(f"⚠️ Aucune feature trouvée pour {model_name}")
            continue

        if end_date is not None and len(dates_all) > 0:
            keep_idx = [i for i, d in enumerate(dates_all) if pd.notna(d) and d <= end_date]
            if len(keep_idx) == 0:
                print(f"⚠️ Aucun modèle {model_name} <= {end_date.date()}")
                continue
            last_i = keep_idx[-1]
        else:
            last_i = len(models_all) - 1

        model_selected = models_all[last_i]

        prep_selected = None
        if isinstance(prep_all, list) and len(prep_all) > last_i:
            prep_selected = prep_all[last_i]
        elif not isinstance(prep_all, list):
            prep_selected = prep_all

        available_feats = [c for c in feats_all if c in ts_base.columns]
        needed_cols = [date_col] + available_feats

        if len(available_feats) == 0:
            print(f"⚠️ Aucune colonne feature disponible dans ts_data pour {model_name}")
            continue

        X_model = ts_base[needed_cols].copy()

        models_dict[model_name] = (model_selected, prep_selected)
        X_dict[model_name] = X_model
        features_dict[model_name] = available_feats

        print(f"✅ {model_name}: model ok | X shape={X_model.shape} | n_features={len(available_feats)}")

    return models_dict, X_dict, features_dict


def _unwrap_model(m):
    base = m

    if hasattr(base, "_base"):
        try:
            base = base._base
        except Exception:
            pass

    try:
        from sklearn.pipeline import Pipeline
        if isinstance(base, Pipeline):
            base = base.steps[-1][1]
    except Exception:
        pass

    if hasattr(base, "models_") and isinstance(getattr(base, "models_"), dict):
        d = base.models_
        if len(d) > 0:
            inner = next(iter(d.values()))
            return _unwrap_model(inner)

    for attr in ["model", "estimator", "_model", "_estimator"]:
        if hasattr(base, attr):
            try:
                inner = getattr(base, attr)
                if inner is not None and inner is not base:
                    return _unwrap_model(inner)
            except Exception:
                pass

    return base


def _name_blob(model) -> str:
    cls = type(model)
    return f"{cls.__name__}|{getattr(cls, '__module__', '')}".lower()


def _is_tree_model(model) -> bool:
    blob = _name_blob(model)
    keys = ("lgbm", "lightgbm", "xgboost", "catboost", "histgradient", "randomforest", "gradientboost")
    return any(k in blob for k in keys)


def _apply_prep(X: pd.DataFrame, prep):
    if prep is None:
        return X

    if hasattr(prep, "transform"):
        Xp = prep.transform(X)
        try:
            return pd.DataFrame(Xp, index=X.index, columns=X.columns)
        except Exception:
            return pd.DataFrame(Xp, index=X.index)

    Xp = X.copy()

    if isinstance(prep, dict):
        lower = prep.get("lower_wins", prep.get("lower", None))
        upper = prep.get("upper_wins", prep.get("upper", None))

        if lower is not None and upper is not None:
            lo = pd.Series(lower, index=Xp.columns) if not isinstance(lower, pd.Series) else lower.reindex(Xp.columns)
            up = pd.Series(upper, index=Xp.columns) if not isinstance(upper, pd.Series) else upper.reindex(Xp.columns)
            Xp = Xp.clip(lower=lo, upper=up, axis=1)

        if prep.get("norm", False):
            mean = prep.get("mean", None)
            std = prep.get("std", None)
            if mean is not None and std is not None:
                mean_s = pd.Series(mean, index=Xp.columns) if not isinstance(mean, pd.Series) else mean.reindex(Xp.columns)
                std_s = pd.Series(std, index=Xp.columns) if not isinstance(std, pd.Series) else std.reindex(Xp.columns)
                Xp = (Xp - mean_s) / std_s.replace(0, 1)

    return Xp


def _lgbm_feature_names(model):
    names = getattr(model, "feature_name_", None)
    if names:
        return list(names)

    booster = getattr(model, "_Booster", None)
    if booster is not None and hasattr(booster, "feature_name"):
        try:
            return list(booster.feature_name())
        except Exception:
            pass

    return None


def _reorder_X_for_lgbm(model, X: pd.DataFrame) -> pd.DataFrame:
    names = _lgbm_feature_names(model)
    if names:
        cols = [c for c in names if c in X.columns]
        if cols:
            return X.loc[:, cols]
    return X


def _compute_phi(model, X: pd.DataFrame, prep=None) -> np.ndarray:
    base = _unwrap_model(model)

    if _is_tree_model(base):
        Xp = _apply_prep(_reorder_X_for_lgbm(base, X), prep).astype(float)

        try:
            expl = shap.TreeExplainer(base, Xp, check_additivity=False)
            sv = expl.shap_values(Xp)
        except TypeError:
            expl = shap.TreeExplainer(base, Xp)
            sv = expl.shap_values(Xp, check_additivity=False)

        phi = sv[0] if isinstance(sv, list) else np.asarray(sv)

        if np.ndim(phi) == 3:
            phi = phi[0]

        if phi.shape[1] != Xp.shape[1]:
            raise ValueError(f"Dimensions SHAP incohérentes: phi={phi.shape}, X={Xp.shape}, model={type(base)}")

        return np.asarray(phi)

    coef = getattr(base, "coef_", None)
    if coef is None:
        raise TypeError(f"Modèle non supporté après unwrap: {type(base)}")

    Xp = _apply_prep(X, prep).astype(float)
    mu = np.nanmean(Xp, axis=0)
    Xc = Xp - mu
    phi = Xc.to_numpy() * np.asarray(coef).reshape(1, -1)
    return phi


def _normalize_feat_name(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _strip_lag_suffix(s: str) -> str:
    s = str(s)
    s = re.sub(r"_lag\d+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_lags?\d+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"lag\d+_", "", s, flags=re.IGNORECASE)
    s = re.sub(r"lags?\d+_", "", s, flags=re.IGNORECASE)
    return s


def _extract_lag_num(s: str):
    s = str(s)
    patterns = [
        r"_lag(\d+)$",
        r"_lags?(\d+)$",
        r"lag(\d+)_",
        r"lags?(\d+)_",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _resolve_feature_name(col_list, feat):
    col_list = list(col_list)
    feat = str(feat)

    if feat in col_list:
        return feat

    feat_norm = _normalize_feat_name(feat)
    feat_base = _normalize_feat_name(_strip_lag_suffix(feat))
    feat_lag = _extract_lag_num(feat)

    for col in col_list:
        if _normalize_feat_name(col) == feat_norm:
            return col

    for col in col_list:
        col_base = _normalize_feat_name(_strip_lag_suffix(col))
        col_lag = _extract_lag_num(col)
        if col_base == feat_base and col_lag == feat_lag:
            return col

    if feat_lag is None:
        same_base = []
        for col in col_list:
            col_base = _normalize_feat_name(_strip_lag_suffix(col))
            if col_base == feat_base:
                same_base.append(col)

        if len(same_base) == 1:
            return same_base[0]

        if len(same_base) > 1:
            same_base_sorted = sorted(
                same_base,
                key=lambda c: (_extract_lag_num(c) is None, _extract_lag_num(c) or 10**9)
            )
            return same_base_sorted[0]

    for col in col_list:
        col_norm = _normalize_feat_name(col)
        if feat_base in col_norm or col_norm in feat_norm:
            return col

    return None


def _ensure_ms_ts(x):
    return pd.Timestamp(x).to_period("M").to_timestamp(how="start").normalize()


def _coerce_to_month_start_index(X: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    out = X.copy()

    if date_col in out.columns:
        ds = pd.to_datetime(out[date_col], errors="coerce")
        ds = ds.dt.to_period("M").dt.to_timestamp(how="start").dt.normalize()
        keep = ds.notna()
        out = out.loc[keep].copy()
        out[date_col] = ds.loc[keep]
        out = out.set_index(date_col)
        return out

    if isinstance(out.index, pd.DatetimeIndex):
        idx = pd.to_datetime(out.index, errors="coerce")
        keep = ~pd.isna(idx)
        out = out.loc[keep].copy()
        out.index = idx[keep].to_period("M").to_timestamp(how="start").normalize()
        return out

    try:
        idx = pd.to_datetime(out.index, errors="coerce")
        if not pd.isna(idx).all():
            keep = ~pd.isna(idx)
            out = out.loc[keep].copy()
            out.index = idx[keep].to_period("M").to_timestamp(how="start").normalize()
            return out
    except Exception:
        pass

    return out


def _filter_X_by_date(X: pd.DataFrame, start_date=None, end_date=None, date_col: str = "ds") -> pd.DataFrame:
    out = _coerce_to_month_start_index(X, date_col=date_col)

    if start_date is None and end_date is None:
        return out

    start = _ensure_ms_ts(start_date) if start_date is not None else None
    end = _ensure_ms_ts(end_date) if end_date is not None else None

    if isinstance(out.index, pd.DatetimeIndex):
        if start is not None:
            out = out.loc[out.index >= start]
        if end is not None:
            out = out.loc[out.index <= end]

    return out


def _pretty_feat_label(feat: str, pretty_map=None):
    if not pretty_map:
        return feat

    if feat in pretty_map:
        return pretty_map[feat]

    base = _strip_lag_suffix(feat)
    lag = _extract_lag_num(feat)

    if base in pretty_map:
        return f"{pretty_map[base]} (lag {lag})" if lag is not None else pretty_map[base]

    return feat


def plot_functional_grid(
    models_dict,
    X_dict,
    selected_vars,
    *,
    start_date=None,
    end_date=None,
    date_col: str = "ds",
    poly_deg_lgbm=3,
    scatter_alpha=0.70,
    s=18,
    colors=_DARK_COLORS,
    figsize_per_cell=(3.1, 2.7),
    sharey=False,
    max_n=2000,
    random_state=42,
    _pretty_label_from_feat=None,
    use_pretty_labels: bool = True,
):
    model_names = list(models_dict.keys())

    if len(model_names) == 0:
        raise ValueError("models_dict est vide. Aucun modèle disponible pour tracer la grille.")
    if len(selected_vars) == 0:
        raise ValueError("selected_vars est vide. Aucune variable à tracer.")

    X_dict_f = {}
    for m in model_names:
        if m not in X_dict:
            raise KeyError(f"X_dict ne contient pas la clé modèle: {m}")

        Xi = _filter_X_by_date(
            X_dict[m],
            start_date=start_date,
            end_date=end_date,
            date_col=date_col,
        )

        if len(Xi) == 0:
            raise RuntimeError(f"Aucune donnée disponible pour le modèle '{m}' après filtre date.")

        X_dict_f[m] = Xi

    n_rows, n_cols = len(selected_vars), len(model_names)
    fig_w = max(6.0, n_cols * figsize_per_cell[0])
    fig_h = max(6.0, n_rows * figsize_per_cell[1])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharey=sharey)
    fig.patch.set_facecolor("#0d0d0d")

    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for c, (mname, (model, prep)) in enumerate(models_dict.items()):
        X_full_all = X_dict_f[mname].copy()

        if max_n is not None and len(X_full_all) > max_n:
            X_full_all = X_full_all.sample(max_n, random_state=random_state).sort_index()

        base = _unwrap_model(model)
        is_linear = hasattr(base, "coef_") and not _is_tree_model(base)

        phi = np.asarray(_compute_phi(model, X_full_all, prep=prep))
        X_plot_all = _apply_prep(X_full_all, prep) if is_linear else X_full_all

        if isinstance(X_plot_all, pd.DataFrame):
            X_plot_all.columns = X_full_all.columns

        col_index_map = {col: j for j, col in enumerate(X_full_all.columns)}
        deg = 1 if is_linear else poly_deg_lgbm
        color_pts = colors[c % len(colors)]

        for r, feat in enumerate(selected_vars):
            ax = axes[r, c]
            ax.set_facecolor("#111111")

            resolved = _resolve_feature_name(X_full_all.columns, feat)
            ylab = _pretty_feat_label(feat, _pretty_label_from_feat) if use_pretty_labels else feat

            if resolved is None or resolved not in col_index_map:
                ax.text(
                    0.5, 0.5, "NA",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color=_DARK_NA,
                    transform=ax.transAxes
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if r == 0:
                    ax.set_title(mname, fontsize=9, pad=4, color="#e5e5e5")
                if c == 0:
                    ax.set_ylabel(ylab, fontsize=9, color="#e5e5e5")
                continue

            j = col_index_map[resolved]
            x = X_plot_all.iloc[:, j].to_numpy()
            y = phi[:, j]

            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]

            ax.scatter(
                x, y,
                alpha=scatter_alpha,
                s=s,
                facecolors="#0d0d0d",
                edgecolors=color_pts,
                linewidths=0.8
            )

            if len(x) > deg + 1:
                try:
                    coefs = np.polyfit(x, y, deg=deg)
                    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                    y_line = np.polyval(coefs, x_line)
                    ax.plot(x_line, y_line, color=_DARK_LINE, linewidth=1.2)
                except Exception:
                    pass

            ax.axhline(0, color=_DARK_ZERO, lw=0.9)
            ax.axvline(0, color=_DARK_ZERO, lw=0.9)
            ax.grid(axis="y", linestyle=_DARK_GRID[1], alpha=_DARK_GRID[0])

            if r == 0:
                ax.set_title(mname, fontsize=9, pad=4, color="#e5e5e5")
            if c == 0:
                ax.set_ylabel(ylab, fontsize=9, color="#e5e5e5")
            if r == n_rows - 1:
                ax.set_xlabel("Observed values", fontsize=8, color="#e5e5e5")

            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#bfbfbf")
            ax.spines["bottom"].set_color("#bfbfbf")

    s_txt = _ensure_ms_ts(start_date).date() if start_date is not None else "min"
    e_txt = _ensure_ms_ts(end_date).date() if end_date is not None else "max"
    fig.suptitle(f"Functional-form grid | {s_txt} → {e_txt}", y=1.01, fontsize=11, color="#e5e5e5")

    plt.tight_layout(h_pad=0.9, w_pad=0.9)
    return fig, axes