import sys
from pathlib import Path
import webbrowser
import json

import mlflow
from mlflow.tracking import MlflowClient

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

# =========================
# PATH SETUP
# =========================
ENTRYPOINT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENTRYPOINT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# =========================
# IMPORTS
# =========================
from ML_pipeline.config.config_loader import load_config
from ML_pipeline.Data_preparation.data_preparation import load_wide_from_feast

# =========================
# CONSTANTS
# =========================
TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "unrate_forecast_model"
MODEL_ALIAS = "champion"
TOP_K = 3


# =========================
# HELPERS
# =========================
def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"✅ Model loaded from {model_uri}")
    return model


def get_registry_run_info():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    return client, mv.run_id, mv.version


def load_config_and_data():
    requirement_path = SRC_ROOT / "ML_pipeline" / "config" / "requirement.json"
    config = load_config(requirement_path)

    df_wide = load_wide_from_feast(
        feature_ref="stationary_value:value",
        series_ids=config["data"]["series_ids"],
        start=config["data"]["start"],
        end=config["data"]["end"],
        freq=config["forecast"]["frequency"],
    )

    df_wide.index = pd.to_datetime(df_wide.index, errors="coerce")
    df_wide = df_wide.sort_index()

    return config, df_wide


def parse_month_input(month_str: str) -> pd.Timestamp:
    try:
        return pd.Period(month_str.strip(), freq="M").to_timestamp(how="start")
    except Exception:
        raise ValueError(
            f"Format invalide: '{month_str}'. Utilise le format YYYY-MM, par ex. 2026-02."
        )


def build_future_feature_frame(df_wide, lags, start_month, end_month):
    lag_list = [lags] if isinstance(lags, int) else list(lags)

    start_date = parse_month_input(start_month)
    end_date = parse_month_input(end_month)

    if start_date > end_date:
        raise ValueError("❌ Le mois de début doit être antérieur ou égal au mois de fin.")

    future_dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    rows = []

    for future_ds in future_dates:
        row = {"forecast_month": future_ds}

        for col in df_wide.columns:
            for lag in lag_list:
                lagged_date = future_ds - pd.DateOffset(months=lag)

                if lagged_date not in df_wide.index:
                    raise ValueError(
                        f"Missing historical data for {col}_lag{lag} "
                        f"at forecast_month={future_ds.strftime('%Y-%m')} "
                        f"(needed {lagged_date.strftime('%Y-%m-%d')})."
                    )

                value = df_wide.loc[lagged_date, col]
                row[f"{col}_lag{lag}"] = value

        rows.append(row)

    X_future = pd.DataFrame(rows)
    X_model = X_future.drop(columns=["forecast_month"]).copy()

    return X_future, X_model


def build_historical_feature_frame(df_wide, lags):
    lag_list = [lags] if isinstance(lags, int) else list(lags)
    max_lag = max(lag_list)

    df_wide = df_wide.copy().sort_index()
    usable_dates = df_wide.index[max_lag:]

    rows = []
    for current_ds in usable_dates:
        row = {"ds": current_ds}
        for col in df_wide.columns:
            for lag in lag_list:
                lagged_date = current_ds - pd.DateOffset(months=lag)
                if lagged_date in df_wide.index:
                    row[f"{col}_lag{lag}"] = df_wide.loc[lagged_date, col]
                else:
                    row[f"{col}_lag{lag}"] = np.nan
        rows.append(row)

    X_hist = pd.DataFrame(rows)
    X_hist = X_hist.dropna().reset_index(drop=True)
    return X_hist


def try_getattr(obj, attr):
    try:
        return getattr(obj, attr)
    except Exception:
        return None


def unwrap_mlflow_to_estimator(pyfunc_model, max_depth=20):
    current = pyfunc_model
    seen = set()

    for _ in range(max_depth):
        if current is None:
            return None

        obj_id = id(current)
        if obj_id in seen:
            break
        seen.add(obj_id)

        if hasattr(current, "coef_") or hasattr(current, "feature_importances_"):
            return current

        if hasattr(current, "feature_importance"):
            return current

        if hasattr(current, "steps") and getattr(current, "steps"):
            current = current.steps[-1][1]
            continue

        candidates = [
            "_model_impl",
            "python_model",
            "model",
            "_model",
            "model_",
            "estimator",
            "estimator_",
            "regressor",
            "regressor_",
            "final_estimator",
            "final_estimator_",
            "best_estimator_",
            "_estimator",
            "_sk_model",
            "sk_model",
            "lgb_model",
            "lgb_model_",
            "booster_",
            "_Booster",
        ]

        moved = False
        for attr in candidates:
            nxt = try_getattr(current, attr)
            if nxt is not None and nxt is not current:
                current = nxt
                moved = True
                break

        if moved:
            continue

        if hasattr(current, "__dict__"):
            for _, val in current.__dict__.items():
                if val is current:
                    continue
                if (
                    hasattr(val, "coef_")
                    or hasattr(val, "feature_importances_")
                    or hasattr(val, "feature_importance")
                ):
                    return val

            for key in [
                "model",
                "_model",
                "model_",
                "estimator",
                "regressor",
                "sk_model",
                "lgb_model",
                "lgb_model_",
                "_Booster",
                "booster_",
            ]:
                if key in current.__dict__ and current.__dict__[key] is not current:
                    current = current.__dict__[key]
                    moved = True
                    break

        if not moved:
            break

    return current


def get_feature_effect_table(pyfunc_model, feature_names):
    est = unwrap_mlflow_to_estimator(pyfunc_model)

    if est is None:
        raise ValueError(
            "Impossible de retrouver l'estimateur sous-jacent depuis le modèle MLflow."
        )

    if hasattr(est, "coef_"):
        raw_w = np.asarray(est.coef_, dtype=float).reshape(-1)[:len(feature_names)]

        if len(raw_w) != len(feature_names):
            raise ValueError(
                f"Incohérence dimensionnelle: {len(raw_w)} poids vs {len(feature_names)} features."
            )

        abs_w = np.abs(raw_w)
        total = abs_w.sum()
        if total == 0 or not np.isfinite(total):
            raise ValueError("Somme des poids invalide.")

        share = abs_w / total
        relation = np.where(
            raw_w > 0, "positive", np.where(raw_w < 0, "negative", "neutral")
        )

        return pd.DataFrame(
            {
                "feature": feature_names,
                "raw_weight": raw_w,
                "abs_weight": abs_w,
                "shap_share": share,
                "relation_to_target": relation,
            }
        ).sort_values("shap_share", ascending=False).reset_index(drop=True)

    if hasattr(est, "feature_importances_"):
        raw_w = np.asarray(est.feature_importances_, dtype=float).reshape(-1)[:len(feature_names)]

        if len(raw_w) != len(feature_names):
            raise ValueError(
                f"Incohérence dimensionnelle: {len(raw_w)} poids vs {len(feature_names)} features."
            )

        abs_w = np.abs(raw_w)
        total = abs_w.sum()
        if total == 0 or not np.isfinite(total):
            raise ValueError("Somme des poids invalide.")

        share = abs_w / total

        return pd.DataFrame(
            {
                "feature": feature_names,
                "raw_weight": raw_w,
                "abs_weight": abs_w,
                "shap_share": share,
                "relation_to_target": "non_directional_importance",
            }
        ).sort_values("shap_share", ascending=False).reset_index(drop=True)

    if hasattr(est, "feature_importance"):
        raw_w = np.asarray(est.feature_importance(), dtype=float).reshape(-1)[:len(feature_names)]

        if len(raw_w) != len(feature_names):
            raise ValueError(
                f"Incohérence dimensionnelle: {len(raw_w)} poids vs {len(feature_names)} features."
            )

        abs_w = np.abs(raw_w)
        total = abs_w.sum()
        if total == 0 or not np.isfinite(total):
            raise ValueError("Somme des poids invalide.")

        share = abs_w / total

        return pd.DataFrame(
            {
                "feature": feature_names,
                "raw_weight": raw_w,
                "abs_weight": abs_w,
                "shap_share": share,
                "relation_to_target": "non_directional_importance",
            }
        ).sort_values("shap_share", ascending=False).reset_index(drop=True)

    raise ValueError(
        f"Impossible d'extraire les importances depuis l'objet {type(est)}."
    )


def compute_local_explanation_table(row_features, feature_effects):
    scored = feature_effects.copy()
    scored["feature_value"] = scored["feature"].map(lambda f: row_features.get(f, np.nan))
    scored["local_score"] = scored["abs_weight"] * scored["feature_value"].abs()

    scored = scored.replace([np.inf, -np.inf], np.nan)
    scored = scored.dropna(subset=["local_score"])
    scored = scored.sort_values("local_score", ascending=False).reset_index(drop=True)

    total_local = scored["local_score"].sum()
    if total_local > 0 and np.isfinite(total_local):
        scored["local_contribution_pct"] = (scored["local_score"] / total_local) * 100.0
    else:
        scored["local_contribution_pct"] = 0.0

    return scored


def build_monthly_explanations(X_future_full, feature_effects, top_k=3):
    out_rows = []

    for _, row in X_future_full.iterrows():
        month_row = {"forecast_month": row["forecast_month"]}
        scored = compute_local_explanation_table(row, feature_effects)
        top = scored.head(top_k).copy()

        for rank in range(len(top)):
            month_row[f"top{rank+1}_feature"] = top.loc[rank, "feature"]
            month_row[f"top{rank+1}_local_contribution_pct"] = round(
                float(top.loc[rank, "local_contribution_pct"]), 2
            )

        out_rows.append(month_row)

    return pd.DataFrame(out_rows)


def build_functional_dataset(model, X_hist_full):
    X_hist = X_hist_full.copy()

    if "ds" in X_hist.columns:
        X_num = X_hist.drop(columns=["ds"]).copy()
    else:
        X_num = X_hist.copy()

    X_num = X_num.apply(pd.to_numeric, errors="coerce")
    X_num = X_num.replace([np.inf, -np.inf], np.nan).dropna().copy()

    est = unwrap_mlflow_to_estimator(model)
    if est is None:
        raise ValueError("Impossible de retrouver l'estimateur pour functional plot.")

    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float).reshape(-1)
        coef = coef[:X_num.shape[1]]

        mu = np.nanmean(X_num.to_numpy(dtype=float), axis=0)
        phi = (X_num.to_numpy(dtype=float) - mu.reshape(1, -1)) * coef.reshape(1, -1)

    elif hasattr(est, "feature_importances_") or hasattr(est, "feature_importance") or hasattr(est, "_Booster"):
        try:
            explainer = shap.TreeExplainer(est)
            shap_values = explainer.shap_values(X_num)

            if isinstance(shap_values, list):
                phi = np.asarray(shap_values[0])
            else:
                phi = np.asarray(shap_values)

            if phi.ndim == 3:
                phi = phi[0]

            if phi.shape[1] != X_num.shape[1]:
                raise ValueError(
                    f"Dimensions incohérentes entre SHAP {phi.shape} et X {X_num.shape}"
                )

        except Exception as e:
            raise ValueError(
                f"Impossible de calculer les valeurs SHAP pour les graphiques de dépendance: {e}"
            )

    else:
        raise ValueError(
            f"Type de modèle non supporté pour la relation fonctionnelle: {type(est)}"
        )

    long_rows = []
    for j, col in enumerate(X_num.columns):
        tmp = pd.DataFrame(
            {
                "feature": col,
                "x_value": X_num[col].to_numpy(dtype=float),
                "effect_on_prediction": phi[:, j],
            }
        )
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
        long_rows.append(tmp)

    functional_df = pd.concat(long_rows, ignore_index=True)
    return functional_df


def build_curve_for_feature(functional_df, feature_name, n_points=120, poly_degree=3):
    sub = functional_df.loc[functional_df["feature"] == feature_name].copy()

    if sub.empty:
        return pd.DataFrame(columns=["x_value", "effect_on_prediction"])

    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["x_value", "effect_on_prediction"]
    )
    sub = sub.sort_values("x_value").reset_index(drop=True)

    if len(sub) < poly_degree + 2 or sub["x_value"].nunique() < poly_degree + 1:
        return sub[["x_value", "effect_on_prediction"]].copy()

    x = sub["x_value"].to_numpy(dtype=float)
    y = sub["effect_on_prediction"].to_numpy(dtype=float)

    try:
        coeffs = np.polyfit(x, y, deg=poly_degree)
        poly = np.poly1d(coeffs)

        x_smooth = np.linspace(np.nanmin(x), np.nanmax(x), n_points)
        y_smooth = poly(x_smooth)

        return pd.DataFrame(
            {
                "x_value": x_smooth,
                "effect_on_prediction": y_smooth,
            }
        )
    except Exception:
        return sub[["x_value", "effect_on_prediction"]].copy()


def _scatter_for_feature(functional_df, feature_name, max_points=250):
    sub = functional_df.loc[
        functional_df["feature"] == feature_name,
        ["x_value", "effect_on_prediction"]
    ].copy()

    if sub.empty:
        return {"x": [], "y": []}

    sub = sub.sort_values("x_value").reset_index(drop=True)

    if len(sub) > max_points:
        idx = np.linspace(0, len(sub) - 1, max_points).astype(int)
        sub = sub.iloc[idx].reset_index(drop=True)

    return {
        "x": [None if pd.isna(v) else float(v) for v in sub["x_value"].tolist()],
        "y": [None if pd.isna(v) else float(v) for v in sub["effect_on_prediction"].tolist()],
    }


def export_plot(final_df, functional_df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = final_df.copy()
    plot_df["forecast_month"] = pd.to_datetime(plot_df["forecast_month"])

    for k in [1, 2, 3]:
        if f"top{k}_feature" not in plot_df.columns:
            plot_df[f"top{k}_feature"] = ""
        if f"top{k}_local_contribution_pct" not in plot_df.columns:
            plot_df[f"top{k}_local_contribution_pct"] = 0.0

        plot_df[f"top{k}_feature"] = plot_df[f"top{k}_feature"].fillna("").astype(str)
        plot_df[f"top{k}_local_contribution_pct"] = pd.to_numeric(
            plot_df[f"top{k}_local_contribution_pct"], errors="coerce"
        ).fillna(0.0)

    first_row = plot_df.iloc[0]

    init_features = [
        first_row["top3_feature"],
        first_row["top2_feature"],
        first_row["top1_feature"],
    ]
    init_values = [
        float(first_row["top3_local_contribution_pct"]),
        float(first_row["top2_local_contribution_pct"]),
        float(first_row["top1_local_contribution_pct"]),
    ]

    init_top1 = first_row["top1_feature"] if first_row["top1_feature"] else ""
    init_top2 = first_row["top2_feature"] if first_row["top2_feature"] else ""
    init_top3 = first_row["top3_feature"] if first_row["top3_feature"] else ""

    init_scatter1 = _scatter_for_feature(functional_df, init_top1)
    init_scatter2 = _scatter_for_feature(functional_df, init_top2)
    init_scatter3 = _scatter_for_feature(functional_df, init_top3)

    init_curve1 = build_curve_for_feature(functional_df, init_top1, poly_degree=3)
    init_curve2 = build_curve_for_feature(functional_df, init_top2, poly_degree=3)
    init_curve3 = build_curve_for_feature(functional_df, init_top3, poly_degree=3)

    fig = make_subplots(
        rows=2,
        cols=6,
        column_widths=[0.17, 0.17, 0.16, 0.16, 0.17, 0.17],
        row_heights=[0.46, 0.54],
        specs=[
            [
                {"type": "scatter", "colspan": 3}, None, None,
                {"type": "bar", "colspan": 3}, None, None,
            ],
            [
                {"type": "scatter", "colspan": 2}, None,
                {"type": "scatter", "colspan": 2}, None,
                {"type": "scatter", "colspan": 2}, None,
            ],
        ],
        subplot_titles=(
            "UNRATE Forecast",
            f"Top 3 explicabilité — {first_row['forecast_month'].strftime('%Y-%m')}",
            "Relation — " + init_top1 if init_top1 else "Relation — Top 1",
            "Relation — " + init_top2 if init_top2 else "Relation — Top 2",
            "Relation — " + init_top3 if init_top3 else "Relation — Top 3",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )

    customdata = np.stack(
        [
            plot_df["top1_feature"],
            plot_df["top1_local_contribution_pct"],
            plot_df["top2_feature"],
            plot_df["top2_local_contribution_pct"],
            plot_df["top3_feature"],
            plot_df["top3_local_contribution_pct"],
        ],
        axis=-1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["forecast_month"],
            y=plot_df["UNRATE_growth_predicted"],
            mode="lines+markers",
            name="Forecast",
            customdata=customdata,
            line=dict(color="#ffffff", width=3),
            marker=dict(color="#ffffff", size=7),
            hovertemplate=(
                "<b>Date prévue</b>: %{x|%Y-%m}<br>"
                "<b>UNRATE growth predicted</b>: %{y:.4f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=init_values,
            y=init_features,
            orientation="h",
            name="Explicabilité",
            marker=dict(color="#ff4d4d"),
            text=[f"{v:.2f}%" for v in init_values],
            textposition="outside",
            hovertemplate=(
                "<b>Variable</b>: %{y}<br>"
                "<b>Contribution locale</b>: %{x:.2f}%<extra></extra>"
            ),
        ),
        row=1,
        col=4,
    )

    fig.add_trace(
        go.Scatter(
            x=init_scatter1["x"],
            y=init_scatter1["y"],
            mode="markers",
            marker=dict(size=6, color="#22ff55", opacity=0.55),
            name="Top1 points",
            hovertemplate="<b>Valeur</b>: %{x:.4f}<br><b>Effet</b>: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=init_curve1["x_value"] if not init_curve1.empty else [],
            y=init_curve1["effect_on_prediction"] if not init_curve1.empty else [],
            mode="lines",
            line=dict(color="#22ff55", width=4),
            name="Top1 curve",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=init_scatter2["x"],
            y=init_scatter2["y"],
            mode="markers",
            marker=dict(size=6, color="#22ff55", opacity=0.55),
            name="Top2 points",
            hovertemplate="<b>Valeur</b>: %{x:.4f}<br><b>Effet</b>: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=init_curve2["x_value"] if not init_curve2.empty else [],
            y=init_curve2["effect_on_prediction"] if not init_curve2.empty else [],
            mode="lines",
            line=dict(color="#22ff55", width=4),
            name="Top2 curve",
            hoverinfo="skip",
        ),
        row=2,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=init_scatter3["x"],
            y=init_scatter3["y"],
            mode="markers",
            marker=dict(size=6, color="#22ff55", opacity=0.55),
            name="Top3 points",
            hovertemplate="<b>Valeur</b>: %{x:.4f}<br><b>Effet</b>: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=5,
    )
    fig.add_trace(
        go.Scatter(
            x=init_curve3["x_value"] if not init_curve3.empty else [],
            y=init_curve3["effect_on_prediction"] if not init_curve3.empty else [],
            mode="lines",
            line=dict(color="#22ff55", width=4),
            name="Top3 curve",
            hoverinfo="skip",
        ),
        row=2,
        col=5,
    )

    fig.update_layout(
        template="plotly_dark",
        title="UNRATE Forecast + Explicabilité locale + Dépendances SHAP Top 3",
        hovermode="closest",
        showlegend=False,
        height=900,
        margin=dict(l=40, r=40, t=90, b=40),
    )

    fig.update_xaxes(title_text="Forecast month", row=1, col=1)
    fig.update_yaxes(title_text="UNRATE growth predicted", row=1, col=1)

    fig.update_xaxes(title_text="Contribution locale (%)", row=1, col=4)
    fig.update_yaxes(title_text="Variables", row=1, col=4)

    fig.update_xaxes(title_text="Observed values", row=2, col=1)
    fig.update_yaxes(title_text="Effect", row=2, col=1)

    fig.update_xaxes(title_text="Observed values", row=2, col=3)
    fig.update_yaxes(title_text="Effect", row=2, col=3)

    fig.update_xaxes(title_text="Observed values", row=2, col=5)
    fig.update_yaxes(title_text="Effect", row=2, col=5)

    plot_path = output_dir / "forecast_unrate_growth_explainability_top3_dependence.html"

    features_payload = {}
    for feat in functional_df["feature"].dropna().astype(str).unique():
        curve = build_curve_for_feature(functional_df, feat, poly_degree=3)
        scatter = _scatter_for_feature(functional_df, feat)

        features_payload[feat] = {
            "scatter_x": [None if pd.isna(v) else float(v) for v in scatter["x"]],
            "scatter_y": [None if pd.isna(v) else float(v) for v in scatter["y"]],
            "curve_x": [None if pd.isna(v) else float(v) for v in curve["x_value"].tolist()],
            "curve_y": [None if pd.isna(v) else float(v) for v in curve["effect_on_prediction"].tolist()],
        }

    features_payload_json = json.dumps(features_payload)

    post_script = rf"""
    var plot = document.getElementById('{{plot_id}}');
    var featureCurves = {features_payload_json};

    function getFeaturePayload(featName) {{
        return featureCurves[featName] || {{
            scatter_x: [],
            scatter_y: [],
            curve_x: [],
            curve_y: []
        }};
    }}

    function updateDependence(traceScatterIdx, traceLineIdx, featName) {{
        var payload = getFeaturePayload(featName);

        Plotly.restyle(
            plot,
            {{
                x: [payload.scatter_x],
                y: [payload.scatter_y],
                marker: [{{color: "#22ff55", opacity: 0.55, size: 6}}]
            }},
            [traceScatterIdx]
        );

        Plotly.restyle(
            plot,
            {{
                x: [payload.curve_x],
                y: [payload.curve_y],
                line: [{{color: "#22ff55", width: 4}}]
            }},
            [traceLineIdx]
        );
    }}

    function updateRightPanels(pt) {{
        var cd = pt.customdata;

        var top1Feat = String(cd[0] || '');
        var top2Feat = String(cd[2] || '');
        var top3Feat = String(cd[4] || '');

        var xvals = [Number(cd[5]), Number(cd[3]), Number(cd[1])];
        var yvals = [top3Feat, top2Feat, top1Feat];
        var texts = [
            xvals[0].toFixed(2) + '%',
            xvals[1].toFixed(2) + '%',
            xvals[2].toFixed(2) + '%'
        ];

        Plotly.restyle(
            plot,
            {{
                x: [xvals],
                y: [yvals],
                text: [texts],
                marker: [{{color: "#ff4d4d"}}]
            }},
            [1]
        );

        updateDependence(2, 3, top1Feat);
        updateDependence(4, 5, top2Feat);
        updateDependence(6, 7, top3Feat);

        var monthLabel = '';
        if (pt.x) {{
            var d = new Date(pt.x);
            if (!isNaN(d.getTime())) {{
                monthLabel = d.toISOString().slice(0, 7);
            }} else {{
                monthLabel = String(pt.x).slice(0, 7);
            }}
        }}

        Plotly.relayout(plot, {{
            'annotations[1].text': 'Top 3 explicabilité — ' + monthLabel,
            'annotations[2].text': 'Relation — ' + top1Feat,
            'annotations[3].text': 'Relation — ' + top2Feat,
            'annotations[4].text': 'Relation — ' + top3Feat
        }});
    }}

    plot.on('plotly_hover', function(data) {{
        if (data.points && data.points.length > 0 && data.points[0].curveNumber === 0) {{
            updateRightPanels(data.points[0]);
        }}
    }});

    plot.on('plotly_click', function(data) {{
        if (data.points && data.points.length > 0 && data.points[0].curveNumber === 0) {{
            updateRightPanels(data.points[0]);
        }}
    }});
    """

    fig.write_html(
        plot_path,
        include_plotlyjs="cdn",
        post_script=post_script,
        full_html=True,
    )
    return plot_path


# =========================
# MAIN
# =========================
def main(start_month, end_month):
    print("\n🚀 Starting inference...\n")

    model = load_model()
    _, run_id, version = get_registry_run_info()

    print(f"✅ Version: {version}")
    print(f"✅ Run ID: {run_id}")

    config, df_wide = load_config_and_data()

    X_future_full, X_model = build_future_feature_frame(
        df_wide=df_wide,
        lags=config["feature_engineering"]["lags"],
        start_month=start_month,
        end_month=end_month,
    )

    preds = model.predict(X_model)

    preds_df = pd.DataFrame(
        {
            "forecast_month": X_future_full["forecast_month"],
            "UNRATE_growth_predicted": preds,
        }
    )

    preds_df["UNRATE_growth_predicted"] = pd.to_numeric(
        preds_df["UNRATE_growth_predicted"], errors="coerce"
    ).round(4)

    feature_effects = get_feature_effect_table(
        pyfunc_model=model,
        feature_names=X_model.columns.tolist(),
    )

    monthly_expl_df = build_monthly_explanations(
        X_future_full=X_future_full,
        feature_effects=feature_effects,
        top_k=TOP_K,
    )

    final_df = preds_df.merge(monthly_expl_df, on="forecast_month", how="left")

    X_hist_full = build_historical_feature_frame(
        df_wide=df_wide,
        lags=config["feature_engineering"]["lags"],
    )

    functional_df = build_functional_dataset(
        model=model,
        X_hist_full=X_hist_full,
    )

    output_dir = PROJECT_ROOT / "outputs"
    plot_path = export_plot(final_df, functional_df, output_dir)

    webbrowser.open_new_tab(plot_path.resolve().as_uri())

    print(f"\n✅ Forecast généré de {start_month} à {end_month}")
    print(f"📊 Plot: {plot_path}")

    return final_df


# =========================
# INPUT UTILISATEUR
# =========================
if __name__ == "__main__":
    print("\n📅 Entrez la période de forecasting au format YYYY-MM")
    start_month = input("➡️ Mois de début : ").strip()
    end_month = input("➡️ Mois de fin : ").strip()

    main(start_month, end_month)