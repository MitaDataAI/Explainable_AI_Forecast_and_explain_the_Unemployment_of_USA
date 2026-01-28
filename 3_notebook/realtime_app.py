# realtime_app.py
# Lancer: streamlit run realtime_app.py

import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

UID_COL    = "series_id"
DS_COL     = "date"
CUTOFF_COL = "cutoff"
Y_COL      = "y_obs"
YHAT_COL   = "y_hat_ar"
LO_COL     = "y_hat_ar_lo_95"
HI_COL     = "y_hat_ar_hi_95"

DEFAULT_PLAY_MS = 250


@st.cache_data
def load_df(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df[DS_COL] = pd.to_datetime(df[DS_COL])
    df[CUTOFF_COL] = pd.to_datetime(df[CUTOFF_COL])
    return df


def prepare(df: pd.DataFrame):
    uid = df[UID_COL].iloc[0]

    obs = (
        df[[UID_COL, DS_COL, Y_COL]]
        .drop_duplicates(subset=[UID_COL, DS_COL])
        .sort_values([UID_COL, DS_COL])
        .reset_index(drop=True)
    )

    cutoffs = (
        df.loc[df[UID_COL] == uid, CUTOFF_COL]
        .drop_duplicates()
        .sort_values()
        .to_list()
    )
    return uid, obs, cutoffs


def make_fig(obs_t: pd.DataFrame, fcst_t: pd.DataFrame, cutoff: pd.Timestamp, uid: str) -> go.Figure:
    fcst_future = fcst_t[fcst_t[DS_COL] > cutoff]
    use_fcst = fcst_future if len(fcst_future) else fcst_t

    fig = go.Figure()

    # Observed
    fig.add_trace(go.Scatter(
        x=obs_t[DS_COL],
        y=obs_t[Y_COL],
        mode="lines",
        name="Unemployment rate (%)"
    ))

    # Forecast mean
    fig.add_trace(go.Scatter(
        x=use_fcst[DS_COL],
        y=use_fcst[YHAT_COL],
        mode="lines",
        name="AutoRegressive (AR)"
    ))

    # PI 95%
    fig.add_trace(go.Scatter(
        x=use_fcst[DS_COL],
        y=use_fcst[LO_COL],
        mode="lines",
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=use_fcst[DS_COL],
        y=use_fcst[HI_COL],
        mode="lines",
        fill="tonexty",
        name="95% Prediction Interval"
    ))

    # ligne verticale "now" (sans add_vline)
    x_now = pd.to_datetime(cutoff).to_pydatetime()

    fig.update_layout(
        shapes=[
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=x_now,
                x1=x_now,
                y0=0,
                y1=1,
                line=dict(dash="dash"),
            )
        ],
        annotations=[
            dict(
                x=x_now,
                y=1.03,
                xref="x",
                yref="paper",
                text=f"t = {pd.to_datetime(cutoff).date()}",
                showarrow=False,
            )
        ],
        height=460,
        title=f"{uid} — pseudo temps réel (observations + prévisions)",
        xaxis_title="Date",
        yaxis_title="UNRATE",
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=80, b=40),
    )

    return fig


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Pseudo temps réel UNRATE", layout="wide")
st.title("Pseudo temps réel : observations + prévisions qui se poursuivent")

default_file = "df_ar_forecasts.parquet"
parquet_path = st.sidebar.text_input("Chemin parquet", value=default_file)

if not Path(parquet_path).exists():
    st.error(
        f"Fichier introuvable: {parquet_path}\n\n"
        "Depuis ton notebook, fais:\n"
        "df_ar_forecasts.to_parquet('df_ar_forecasts.parquet', index=False)"
    )
    st.stop()

df = load_df(parquet_path)

required = {UID_COL, DS_COL, CUTOFF_COL, Y_COL, YHAT_COL, LO_COL, HI_COL}
missing = sorted(list(required - set(df.columns)))
if missing:
    st.error(f"Colonnes manquantes dans le parquet: {missing}")
    st.stop()

uid, obs, cutoffs = prepare(df)
if len(cutoffs) == 0:
    st.error("Aucun cutoff trouvé.")
    st.stop()

# ✅ session state (on n'utilise PAS key="i" pour un widget)
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "playing" not in st.session_state:
    st.session_state.playing = False
if "play_ms" not in st.session_state:
    st.session_state.play_ms = DEFAULT_PLAY_MS

with st.sidebar:
    st.header("Contrôles")

    st.session_state.play_ms = st.slider(
        "Vitesse (ms / pas)",
        50, 1500,
        st.session_state.play_ms,
        step=50
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Play", use_container_width=True):
            st.session_state.playing = True
    with c2:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state.playing = False

    c3, c4, c5 = st.columns(3)
    with c3:
        if st.button("⟲ Reset", use_container_width=True):
            st.session_state.idx = 0
            st.session_state.playing = False
    with c4:
        if st.button("◀ Prev", use_container_width=True):
            st.session_state.idx = max(0, st.session_state.idx - 1)
    with c5:
        if st.button("Next ▶", use_container_width=True):
            st.session_state.idx = min(len(cutoffs) - 1, st.session_state.idx + 1)

    # ✅ slider sans key="i" et sans réaffectation directe
    st.session_state.idx = st.slider(
        "Aller au cutoff",
        0, len(cutoffs) - 1,
        st.session_state.idx
    )

# ----------------------------
# Render
# ----------------------------
cutoff = cutoffs[st.session_state.idx]

obs_t = obs[(obs[UID_COL] == uid) & (obs[DS_COL] <= cutoff)]
fcst_t = df[(df[UID_COL] == uid) & (df[CUTOFF_COL] == cutoff)].sort_values(DS_COL)

m1, m2, m3 = st.columns(3)
m1.metric("Cutoff (t)", str(pd.to_datetime(cutoff).date()))
m2.metric("Points observés affichés", int(len(obs_t)))
m3.metric("Points forecast à t", int(len(fcst_t)))

fig = make_fig(obs_t, fcst_t, cutoff, uid)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Autoplay (pseudo streaming)
# ----------------------------
if st.session_state.playing:
    if st.session_state.idx < len(cutoffs) - 1:
        time.sleep(st.session_state.play_ms / 1000.0)
        st.session_state.idx += 1
        st.rerun()
    else:
        st.session_state.playing = False
        st.info("Fin : dernier cutoff atteint.")