import numpy as np
import pandas as pd

from utilsforecast.plotting import plot_series

MODEL_COLORS = {
    "AR": "#2ecc71",
    "LR": "#f39c12",
    "RIDGE": "#3498db",
    "LGBM": "#9b59b6",
}


def plot_forecast_evolution(
    combined_bkt: pd.DataFrame,
    *,
    series_name: str = "Target",
    partition: str | None = None,
    models: tuple = ("AR", "LR", "RIDGE"),
    title_suffix: str = "",
    show_partition_lines: bool = True,
    partition_line_kwargs: dict | None = None,
    annotate_partition_metrics: bool = True,
    annotate_fontsize: int = 10,
    annotate_bbox_alpha: float = 0.25,
):
    """
    Graphique général observed vs forecasts avec séparateurs de partitions
    et annotations de métriques par période.

    Paramètres attendus dans combined_bkt
    ------------------------------------
    Colonnes minimales :
    - ds
    - model_label
    - y_true
    - y_hat

    Colonnes optionnelles :
    - partition
    - covered / width
    - ou y_lo / y_hi

    Retour
    ------
    fig : plotly figure
    metrics_by_part : pd.DataFrame
    """
    if partition_line_kwargs is None:
        partition_line_kwargs = dict(
            line_color="gray",
            line_width=1,
            line_dash="dash",
            opacity=0.6,
        )

    # ======================
    # 0) PREP DATA
    # ======================
    df_all = combined_bkt.copy()
    df_all["ds"] = (
        pd.to_datetime(df_all["ds"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp(how="start")
    )
    df_all = df_all.dropna(subset=["ds"]).copy()

    df_all["model_label"] = df_all["model_label"].astype(str).str.upper().str.strip()
    models_u = tuple(m.upper() for m in models)
    df_all = df_all[df_all["model_label"].isin(models_u)].copy()
    df_all = df_all.dropna(subset=["y_true", "y_hat"]).sort_values(["ds", "model_label"])

    df = df_all.copy()
    if partition is not None and "partition" in df.columns:
        df = df[df["partition"].astype(str) == str(partition)].copy()

    part_label = "ALL" if partition is None else str(partition)

    # ======================
    # 1) METRICS HELPERS
    # ======================
    def ensure_cov_width(tmp: pd.DataFrame) -> pd.DataFrame:
        tmp = tmp.copy()
        tmp["abs_err"] = (tmp["y_true"] - tmp["y_hat"]).abs()

        need_cov = ("covered" not in tmp.columns) or tmp["covered"].isna().all()
        need_wid = ("width" not in tmp.columns) or tmp["width"].isna().all()

        if (need_cov or need_wid) and {"y_lo", "y_hi"}.issubset(tmp.columns):
            lohi = np.sort(tmp[["y_lo", "y_hi"]].to_numpy(), axis=1)
            tmp["y_lo"] = lohi[:, 0]
            tmp["y_hi"] = lohi[:, 1]
            if need_cov:
                tmp["covered"] = (
                    (tmp["y_true"] >= tmp["y_lo"]) & (tmp["y_true"] <= tmp["y_hi"])
                ).astype(float)
            if need_wid:
                tmp["width"] = (tmp["y_hi"] - tmp["y_lo"]).astype(float)

        if "covered" not in tmp.columns:
            tmp["covered"] = np.nan
        if "width" not in tmp.columns:
            tmp["width"] = np.nan

        return tmp

    df_for_part_metrics = ensure_cov_width(df_all)

    if "partition" not in df_for_part_metrics.columns:
        df_for_part_metrics["partition"] = "ALL"
    else:
        df_for_part_metrics["partition"] = df_for_part_metrics["partition"].astype(str)

    metrics_by_part = (
        df_for_part_metrics.groupby(["partition", "model_label"], dropna=False)
        .agg(
            mae=("abs_err", "mean"),
            coverage=("covered", "mean"),
            width=("width", "mean"),
        )
        .reset_index()
    )

    # ======================
    # 2) PARTITION BOUNDARIES
    # ======================
    partition_dates = []
    partition_labels = []

    if (partition is None) and ("partition" in df_all.columns):
        part_series = df_all.groupby("ds")["partition"].first().sort_index()
        changes = part_series[part_series != part_series.shift(1)]
        partition_dates = list(changes.index)
        partition_labels = [str(part_series.loc[d]) for d in partition_dates]

    # ======================
    # 3) GRAPH — OBSERVED VS FORECASTS
    # ======================
    unique_id = str(series_name)

    df_obs = (
        df.groupby("ds")["y_true"].first()
        .rename("y")
        .reset_index()
        .assign(unique_id=unique_id)[["unique_id", "ds", "y"]]
    )

    fcst_parts = []
    for m in models_u:
        sub = df[df["model_label"] == m].copy()
        if sub.empty:
            continue

        out = sub[["ds", "y_hat"]].copy()
        out = out.rename(columns={"y_hat": m})
        out["unique_id"] = unique_id
        fcst_parts.append(out[["unique_id", "ds", m]])

    if len(fcst_parts) == 0:
        raise ValueError("Aucune forecast trouvée pour les modèles demandés dans combined_bkt.")

    df_fcst = fcst_parts[0]
    for k in fcst_parts[1:]:
        df_fcst = df_fcst.merge(k, on=["unique_id", "ds"], how="outer")
    df_fcst = df_fcst.sort_values("ds")

    fig = plot_series(
        df=df_obs,
        forecasts_df=df_fcst,
        engine="plotly",
    ).update_layout(
        height=520,
        template="plotly_dark",
        title=f"{series_name} — Observed vs Forecasts — {part_label}" + (f" {title_suffix}" if title_suffix else ""),
        legend_title_text="",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    for tr in fig.data:
        name = (tr.name or "")
        if name in MODEL_COLORS:
            tr.line.color = MODEL_COLORS[name]
            tr.line.width = 3.0 if name == "AR" else 2.2
        if name == "y":
            tr.line.color = "white"
            tr.line.width = 2.8

    for tr in fig.data:
        n = (tr.name or "")
        if n == "y":
            tr.name = f"{series_name} observed"
        elif n == "LR":
            tr.name = "Linear Regression (exog)"
        elif n == "RIDGE":
            tr.name = "Ridge (exog)"
        elif n == "AR":
            tr.name = "AutoRegressive (AR)"
        elif n == "LGBM":
            tr.name = "LightGBM (exog)"

    if show_partition_lines and (partition is None) and len(partition_dates) > 1:
        for d in partition_dates[1:]:
            fig.add_vline(
                x=pd.to_datetime(d),
                line_color=partition_line_kwargs.get("line_color", "gray"),
                line_width=partition_line_kwargs.get("line_width", 1),
                line_dash=partition_line_kwargs.get("line_dash", "dash"),
                opacity=partition_line_kwargs.get("opacity", 0.6),
            )

    if annotate_partition_metrics and (partition is None) and len(partition_dates) >= 1:
        segs = []
        for i, start in enumerate(partition_dates):
            end = partition_dates[i + 1] if i < len(partition_dates) - 1 else df_obs["ds"].max()
            label = partition_labels[i] if i < len(partition_labels) else "?"
            segs.append((pd.to_datetime(start), pd.to_datetime(end), str(label)))

        for start, end, plabel in segs:
            mid = start + (end - start) / 2
            mbp = metrics_by_part[metrics_by_part["partition"] == plabel]

            lines = [f"<b>{plabel}</b>"]
            for m in models_u:
                row = mbp[mbp["model_label"] == m]
                if len(row):
                    mae = float(row["mae"].iloc[0])
                    cov = row["coverage"].iloc[0]
                    wid = row["width"].iloc[0]
                    cov_txt = "NA" if pd.isna(cov) else f"{float(cov):.2f}"
                    wid_txt = "NA" if pd.isna(wid) else f"{float(wid):.2f}"
                    lines.append(f"{m}: MAE {mae:.3f} | C {cov_txt} | W {wid_txt}")
                else:
                    lines.append(f"{m}: (no data)")

            fig.add_annotation(
                x=mid,
                y=1.02,
                xref="x",
                yref="paper",
                text="<br>".join(lines),
                showarrow=False,
                align="center",
                font=dict(size=annotate_fontsize, color="white"),
                bgcolor=f"rgba(0,0,0,{annotate_bbox_alpha})",
                bordercolor="rgba(255,255,255,0.0)",
                borderpad=6,
            )

    fig.show()
    return fig, metrics_by_part