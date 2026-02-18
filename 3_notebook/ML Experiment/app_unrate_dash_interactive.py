# ============================================================
# Dash (Plotly) — UNRATE + PI95 + C(t) + metrics (comme ton PyQtGraph)
# ✅ Interactif:
# - Hover: tooltip (date, y, ridge, PI95, C)
# - Click (gauche): freeze / unfreeze du curseur (toggle)
# - Bouton "Unfreeze"
# - Sélection période: via ZOOM (drag) => métriques sur la fenêtre visible (RMSE + Coverage)
# - Toggles: True / Pred / Interval
#
# Données attendues: bkt_ridge_final (DataFrame) avec colonnes:
# ['unique_id','ds','y','RIDGE','RIDGE-lo-95','RIDGE-hi-95','alpha_used' (C)]
# ============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update

# ----------------------------
# 0) CONFIG
# ----------------------------
PORT = 8053
DEFAULT_UID = "UNRATE"
C_COL = "alpha_used"  # ton "C"
SHOW_RANGE_SLIDER = True

# ----------------------------
# 1) PREP DATA
# ----------------------------
d = bkt_ridge_final.copy()

required = {"ds", "y", "RIDGE", "RIDGE-lo-95", "RIDGE-hi-95"}
missing = sorted(list(required - set(d.columns)))
if missing:
    raise KeyError(f"Colonnes manquantes: {missing}. Colonnes dispo: {list(d.columns)}")

if "unique_id" not in d.columns:
    d["unique_id"] = DEFAULT_UID

d["ds"] = pd.to_datetime(d["ds"])
d = d.sort_values(["unique_id", "ds"]).reset_index(drop=True)

HAS_C = C_COL in d.columns
unique_ids = sorted(d["unique_id"].dropna().unique().tolist())
default_uid = DEFAULT_UID if DEFAULT_UID in unique_ids else unique_ids[0]

# ----------------------------
# 2) METRICS
# ----------------------------
def rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

def coverage95(y, lo, hi) -> float:
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(m):
        return float("nan")
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))

def _format(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "NaN"
    return f"{v:.{nd}f}"

def _format_c(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "NaN"
    return f"{v:.4g}"

# ----------------------------
# 3) FIG BUILDERS
# ----------------------------
def make_main_fig(df: pd.DataFrame, toggles: set, cursor_dt=None):
    fig = go.Figure()

    # Interval (zone)
    if "I" in toggles:
        fig.add_trace(go.Scatter(
            x=df["ds"], y=df["RIDGE-hi-95"],
            name="Ridge 95% Prediction Interval",
            mode="lines", line={"width": 0},
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=df["ds"], y=df["RIDGE-lo-95"],
            name="Ridge 95% Prediction Interval",
            mode="lines", line={"width": 0},
            fill="tonexty",
            hoverinfo="skip",
            showlegend=False
        ))

    # Pred
    if "P" in toggles:
        fig.add_trace(go.Scatter(
            x=df["ds"], y=df["RIDGE"],
            name="Ridge Regression (exog)",
            mode="lines",
        ))

    # True
    if "T" in toggles:
        fig.add_trace(go.Scatter(
            x=df["ds"], y=df["y"],
            name="Unemployment rate (%)" if df["unique_id"].iloc[0] == "UNRATE" else "Observed",
            mode="lines",
        ))

    # cursor vertical line
    shapes = []
    if cursor_dt is not None:
        shapes.append(dict(
            type="line",
            xref="x", yref="paper",
            x0=cursor_dt, x1=cursor_dt,
            y0=0, y1=1,
            line=dict(width=1, dash="dash")
        ))

    fig.update_layout(
        template="plotly_white",
        height=520,
        hovermode="x unified",
        margin={"l": 40, "r": 20, "t": 55, "b": 35},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        shapes=shapes,
    )

    if SHOW_RANGE_SLIDER:
        fig.update_xaxes(rangeslider_visible=True)
    return fig


def make_c_fig(df: pd.DataFrame, cursor_dt=None):
    fig = go.Figure()

    shapes = []
    if cursor_dt is not None:
        shapes.append(dict(
            type="line",
            xref="x", yref="paper",
            x0=cursor_dt, x1=cursor_dt,
            y0=0, y1=1,
            line=dict(width=1, dash="dash")
        ))

    if not HAS_C:
        fig.update_layout(
            template="plotly_white",
            height=260,
            margin={"l": 40, "r": 20, "t": 40, "b": 35},
            title=f"C(t) — colonne '{C_COL}' absente",
            shapes=shapes,
        )
        return fig

    # pour bien voir les changements, on trace en "steps" (hv)
    dfc = df[["ds", C_COL]].dropna().sort_values("ds")
    fig.add_trace(go.Scatter(
        x=dfc["ds"], y=dfc[C_COL],
        name="C(t)",
        mode="lines+markers",
        line_shape="hv",
    ))

    fig.update_layout(
        template="plotly_white",
        height=260,
        hovermode="x unified",
        margin={"l": 40, "r": 20, "t": 40, "b": 35},
        title="C(t) evolution",
        shapes=shapes,
    )
    fig.update_yaxes(title="C")
    return fig

# ----------------------------
# 4) DASH APP
# ----------------------------
app = Dash(__name__)
app.title = "UNRATE — Ridge + PI95 + C(t)"

app.layout = html.Div(
    style={"maxWidth": "1150px", "margin": "18px auto", "fontFamily": "Arial"},
    children=[
        html.H3("Ridge — Forecast + PI95 + C(t) + metrics (Dash)"),

        html.Div(
            style={"display": "flex", "gap": "14px", "alignItems": "center", "flexWrap": "wrap"},
            children=[
                html.Div("Series:", style={"minWidth": "55px"}),
                dcc.Dropdown(
                    id="uid",
                    options=[{"label": u, "value": u} for u in unique_ids],
                    value=default_uid,
                    clearable=False,
                    style={"width": "260px"},
                ),

                html.Div("Toggles:", style={"minWidth": "60px", "marginLeft": "10px"}),
                dcc.Checklist(
                    id="toggles",
                    options=[
                        {"label": "True (T)", "value": "T"},
                        {"label": "Pred (P)", "value": "P"},
                        {"label": "Interval (I)", "value": "I"},
                    ],
                    value=["T", "P", "I"],
                    inline=True,
                ),

                html.Button("Unfreeze", id="btn_unfreeze", n_clicks=0, style={"marginLeft": "10px"}),
                html.Span(id="freeze_state", style={"marginLeft": "10px", "color": "#444"}),
            ],
        ),

        html.Div(
            id="metrics",
            style={"marginTop": "10px", "fontSize": "14px", "color": "#222"},
        ),
        html.Div(
            id="metrics_sel",
            style={"marginTop": "4px", "fontSize": "13px", "color": "#444"},
        ),

        dcc.Graph(id="g_main", clear_on_unhover=False),
        dcc.Graph(id="g_c", clear_on_unhover=False),

        # Stores: curseur + état freeze
        dcc.Store(id="st_freeze", data={"frozen": False}),
        dcc.Store(id="st_cursor", data={"ds": None}),  # date du curseur (hover/click)
    ],
)

# ----------------------------
# 5) CALLBACK: freeze toggle + cursor update
# ----------------------------
@app.callback(
    Output("st_freeze", "data"),
    Output("st_cursor", "data"),
    Output("freeze_state", "children"),
    Input("g_main", "clickData"),          # click => toggle freeze
    Input("g_main", "hoverData"),          # hover => move cursor (si pas frozen)
    Input("btn_unfreeze", "n_clicks"),     # bouton => unfreeze
    State("st_freeze", "data"),
    State("st_cursor", "data"),
    prevent_initial_call=True,
)
def update_cursor(clickData, hoverData, n_unfreeze, st_freeze, st_cursor):
    frozen = bool(st_freeze.get("frozen", False))
    cursor_ds = st_cursor.get("ds", None)

    ctx = app.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update

    trig = ctx.triggered[0]["prop_id"]

    # Unfreeze button
    if trig == "btn_unfreeze.n_clicks":
        return {"frozen": False}, {"ds": cursor_ds}, "State: unfrozen"

    # Click toggles freeze; also set cursor to clicked x
    if trig == "g_main.clickData":
        # extract x
        new_ds = None
        if clickData and clickData.get("points"):
            new_ds = clickData["points"][0].get("x", None)
        # toggle frozen
        new_frozen = not frozen
        # if click has x, update cursor
        if new_ds is not None:
            cursor_ds = new_ds
        return {"frozen": new_frozen}, {"ds": cursor_ds}, f"State: {'frozen' if new_frozen else 'unfrozen'}"

    # Hover updates cursor only if not frozen
    if trig == "g_main.hoverData":
        if frozen:
            return no_update, no_update, "State: frozen"
        new_ds = None
        if hoverData and hoverData.get("points"):
            new_ds = hoverData["points"][0].get("x", None)
        if new_ds is None:
            return no_update, no_update, "State: unfrozen"
        return {"frozen": False}, {"ds": new_ds}, "State: unfrozen"

    return no_update, no_update, no_update


# ----------------------------
# 6) CALLBACK: build figures + metrics (global + visible range)
# - Visible range récupéré via relayoutData (zoom)
# ----------------------------
@app.callback(
    Output("g_main", "figure"),
    Output("g_c", "figure"),
    Output("metrics", "children"),
    Output("metrics_sel", "children"),
    Input("uid", "value"),
    Input("toggles", "value"),
    Input("st_cursor", "data"),
    Input("g_main", "relayoutData"),  # zoom/pan/range
)
def render(uid, toggles, st_cursor, relayoutData):
    if uid is None:
        return no_update, no_update, "", ""

    toggles = set(toggles or [])
    df = d[d["unique_id"] == uid].copy()

    # cursor date (string iso) => keep as is (plotly handles)
    cursor_ds = (st_cursor or {}).get("ds", None)

    # --- Global metrics ---
    g_rmse = rmse(df["y"].values, df["RIDGE"].values)
    g_cov = coverage95(df["y"].values, df["RIDGE-lo-95"].values, df["RIDGE-hi-95"].values)

    c_now = None
    if HAS_C and cursor_ds is not None:
        # nearest match by date
        dt = pd.to_datetime(cursor_ds)
        i = int(np.argmin(np.abs((df["ds"].values - np.datetime64(dt)).astype("timedelta64[D]").astype(int))))
        c_now = float(df.iloc[i][C_COL]) if pd.notna(df.iloc[i][C_COL]) else None

    metrics_txt = (
        f"GLOBAL — RMSE: {_format(g_rmse, 4)} | Coverage(95%): {_format(g_cov, 3)}"
        + (f" | C(at cursor): {_format_c(c_now)}" if HAS_C else "")
    )

    # --- Selection metrics: fenêtre visible (zoom) ---
    x0 = x1 = None
    if isinstance(relayoutData, dict):
        # plotly fournit soit xaxis.range[0]/[1], soit xaxis.range (liste), soit xaxis.autorange
        if "xaxis.range[0]" in relayoutData and "xaxis.range[1]" in relayoutData:
            x0, x1 = relayoutData["xaxis.range[0]"], relayoutData["xaxis.range[1]"]
        elif "xaxis.range" in relayoutData and isinstance(relayoutData["xaxis.range"], (list, tuple)) and len(relayoutData["xaxis.range"]) == 2:
            x0, x1 = relayoutData["xaxis.range"][0], relayoutData["xaxis.range"][1]
        elif relayoutData.get("xaxis.autorange", False):
            x0 = x1 = None

    if x0 is not None and x1 is not None:
        dt0 = pd.to_datetime(x0)
        dt1 = pd.to_datetime(x1)
        dsel = df[(df["ds"] >= dt0) & (df["ds"] <= dt1)].copy()
        s_rmse = rmse(dsel["y"].values, dsel["RIDGE"].values)
        s_cov = coverage95(dsel["y"].values, dsel["RIDGE-lo-95"].values, dsel["RIDGE-hi-95"].values)
        sel_txt = f"VISIBLE WINDOW ({dt0.strftime('%Y-%m')} → {dt1.strftime('%Y-%m')}) — RMSE: {_format(s_rmse,4)} | Coverage(95%): {_format(s_cov,3)} | Rows: {len(dsel)}"
    else:
        sel_txt = "VISIBLE WINDOW — (auto) zoom/pan sur le graphique pour calculer RMSE/Coverage sur la fenêtre visible."

    # --- Build figures (shared cursor line) ---
    f_main = make_main_fig(df, toggles, cursor_ds)
    f_main.update_layout(title=f"{uid} — Ridge forecast (95% PI)")

    f_c = make_c_fig(df, cursor_ds)

    # Synchroniser le zoom entre les deux graphes (même range si présent)
    if x0 is not None and x1 is not None:
        f_c.update_xaxes(range=[x0, x1])

    return f_main, f_c, metrics_txt, sel_txt


# ----------------------------
# 7) RUN
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=PORT, use_reloader=False)