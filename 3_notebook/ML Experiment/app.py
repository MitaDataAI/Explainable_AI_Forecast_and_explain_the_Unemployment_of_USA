import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st


# ============================================================
# Streamlit : plein écran sans UI
# ============================================================
st.set_page_config(layout="wide", page_title="UNRATE — RIDGE + Conformal (stream)", initial_sidebar_state="collapsed")
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
.block-container {padding: 0; margin: 0;}
section[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Chargement (cache)
# ============================================================
DATA_PATH = Path("outputs/streamlit/bkt_ridge_final.parquet")

def _find_pi_cols(df: pd.DataFrame, model_col: str) -> tuple[str, str]:
    """
    Try to find conformal PI 95% columns.
    Priority:
      1) f"{model}-lo-95" / f"{model}-hi-95"
      2) generic search with (lo|lower) & 95 & model name
    """
    direct_lo = f"{model_col}-lo-95"
    direct_hi = f"{model_col}-hi-95"
    if direct_lo in df.columns and direct_hi in df.columns:
        return direct_lo, direct_hi

    cols = list(df.columns)
    m = model_col.lower()

    lo_candidates = [
        c for c in cols
        if ("lo" in c.lower() or "lower" in c.lower())
        and ("95" in c)
        and (m in c.lower() or "level" in c.lower())
    ]
    hi_candidates = [
        c for c in cols
        if ("hi" in c.lower() or "upper" in c.lower())
        and ("95" in c)
        and (m in c.lower() or "level" in c.lower())
    ]
    if lo_candidates and hi_candidates:
        return lo_candidates[0], hi_candidates[0]

    raise KeyError(
        f"Impossible de trouver les colonnes PI 95% pour {model_col}. "
        f"Attendu: '{model_col}-lo-95' et '{model_col}-hi-95' (ou équivalent)."
    )

@st.cache_data(show_spinner=False)
def load_bkt(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path.resolve()}")

    df = pd.read_parquet(path)
    if "ds" not in df.columns:
        raise KeyError(f"Missing 'ds'. Columns: {list(df.columns)}")

    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    # true
    if "y_true" in df.columns:
        y_true_col = "y_true"
    elif "y" in df.columns:
        y_true_col = "y"
    else:
        raise KeyError("Missing true column: need 'y_true' or 'y'.")

    # ridge
    if "RIDGE" in df.columns:
        ridge_col = "RIDGE"
    elif "Ridge" in df.columns:
        ridge_col = "Ridge"
    else:
        raise KeyError("Missing prediction column: need 'RIDGE' or 'Ridge'.")

    # conformal PI 95%
    lo_col, hi_col = _find_pi_cols(df, ridge_col)

    out = df[["ds", y_true_col, ridge_col, lo_col, hi_col]].rename(
        columns={
            y_true_col: "y_true",
            ridge_col: "y_ridge",
            lo_col: "y_lo",
            hi_col: "y_hi",
        }
    )
    out = out.dropna(subset=["ds"]).reset_index(drop=True)
    return out

try:
    d = load_bkt(str(DATA_PATH))
except Exception as e:
    st.error("Impossible de charger bkt_ridge_final.parquet")
    st.code(str(e))
    st.stop()


# ============================================================
# Réglages fluidité
# ============================================================
TARGET_FPS   = 120
REDRAW_EVERY = 1.0 / TARGET_FPS
WINDOW       = 600
MAX_PTS      = 900
SMOOTH_SPAN  = 1
JPEG_QUALITY = 70
DPI          = 90
LINE_SCALE   = 2.6


# ============================================================
# Pré-calculs (✅ idx en DatetimeIndex)
# ============================================================
idx = pd.DatetimeIndex(d["ds"])
x_full_num = mdates.date2num(idx.to_pydatetime()).astype(np.float64)

def smooth_ema_arr(arr: np.ndarray, span: int) -> np.ndarray:
    s = pd.Series(arr)
    if span and span > 1:
        return s.ewm(span=span, adjust=False).mean().astype(np.float32).values
    return s.astype(np.float32).values

y_true_all  = smooth_ema_arr(d["y_true"].to_numpy(dtype=np.float32),  SMOOTH_SPAN)
y_ridge_all = smooth_ema_arr(d["y_ridge"].to_numpy(dtype=np.float32), SMOOTH_SPAN)
y_lo_all    = smooth_ema_arr(d["y_lo"].to_numpy(dtype=np.float32),    SMOOTH_SPAN)
y_hi_all    = smooth_ema_arr(d["y_hi"].to_numpy(dtype=np.float32),    SMOOTH_SPAN)

valid_true_mask  = ~np.isnan(y_true_all)
valid_ridge_mask = ~np.isnan(y_ridge_all)


# ============================================================
# Figure persistante
# ============================================================
fig, ax = plt.subplots(figsize=(22, 10), dpi=DPI)

fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.grid(alpha=0.25)

ax.set_xlabel("Time", color="white")
ax.set_ylabel("Value", color="white")
ax.tick_params(colors="white")

ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

ax.set_title("UNRATE forecasting (stream)\nRMSE RIDGE=... · Coverage(95%)=...", color="white", fontsize=18)

(line_true,)  = ax.plot([], [], linewidth=1.8 * LINE_SCALE, label="UNRATE (true)",  color="#87CEEB")
(line_ridge,) = ax.plot([], [], linewidth=2.0 * LINE_SCALE, linestyle="--", label="RIDGE (pred)", color="orange")

# PI boundary lines (optional but nice)
(line_lo,) = ax.plot([], [], linewidth=1.2 * LINE_SCALE, linestyle=":", label="PI 95% (lo)", color="#aaaaaa")
(line_hi,) = ax.plot([], [], linewidth=1.2 * LINE_SCALE, linestyle=":", label="PI 95% (hi)", color="#aaaaaa")

# band handle (we will recreate it each redraw)
band = None

vline = ax.axvline(
    x_full_num[0],
    linestyle="--",
    linewidth=1.0 * LINE_SCALE,
    alpha=0.5,
    color="#666666"
)

base_size = plt.rcParams.get("legend.fontsize", 10)
if isinstance(base_size, str):
    base_size = 10
leg = ax.legend(loc="upper left", fontsize=3 * base_size)
leg.get_frame().set_facecolor("black")
leg.get_frame().set_edgecolor("white")
for t in leg.get_texts():
    t.set_color("white")

ph_text = st.empty()
ph_fig  = st.empty()


# ============================================================
# Helpers
# ============================================================
def window_slice(i: int):
    start = max(0, i - WINDOW)
    stop = i
    return start, stop

def downsample_step(n: int, max_pts: int):
    if n <= max_pts:
        return None
    return max(1, n // max_pts)


# ============================================================
# Metrics incrémentales
# ============================================================
sse = 0.0
cnt = 0

# coverage 95% : proportion de y_true dans [lo, hi]
inside = 0
cnt_cov = 0

last_draw = 0.0
N = len(x_full_num)

for i in range(1, N + 1):
    k = i - 1

    # RMSE incrémental
    if valid_true_mask[k] and valid_ridge_mask[k]:
        diff = float(y_true_all[k] - y_ridge_all[k])
        if diff == diff:
            sse += diff * diff
            cnt += 1

    # Coverage incrémentale
    yt = y_true_all[k]
    lo = y_lo_all[k]
    hi = y_hi_all[k]
    if (yt == yt) and (lo == lo) and (hi == hi):
        cnt_cov += 1
        if lo <= yt <= hi:
            inside += 1

    now = time.time()
    if (now - last_draw) < REDRAW_EVERY and i < N:
        continue
    last_draw = now

    # Fenêtre glissante
    start, stop = window_slice(i)
    x_win  = x_full_num[start:stop]
    yt_win = y_true_all[start:stop]
    yr_win = y_ridge_all[start:stop]
    lo_win = y_lo_all[start:stop]
    hi_win = y_hi_all[start:stop]

    # Downsample
    n = stop - start
    step = downsample_step(n, MAX_PTS)
    if step is None:
        x_sel, yt_sel, yr_sel, lo_sel, hi_sel = x_win, yt_win, yr_win, lo_win, hi_win
    else:
        x_sel  = x_win[::step]
        yt_sel = yt_win[::step]
        yr_sel = yr_win[::step]
        lo_sel = lo_win[::step]
        hi_sel = hi_win[::step]
        if x_sel[-1] != x_win[-1]:
            x_sel  = np.concatenate((x_sel,  x_win[-1:]))
            yt_sel = np.concatenate((yt_sel, yt_win[-1:]))
            yr_sel = np.concatenate((yr_sel, yr_win[-1:]))
            lo_sel = np.concatenate((lo_sel, lo_win[-1:]))
            hi_sel = np.concatenate((hi_sel, hi_win[-1:]))

    # Update lines
    line_true.set_data(x_sel, yt_sel)
    line_ridge.set_data(x_sel, yr_sel)
    line_lo.set_data(x_sel, lo_sel)
    line_hi.set_data(x_sel, hi_sel)
    vline.set_xdata([x_sel[-1], x_sel[-1]])

    # Update band (remove & recreate)
    if band is not None:
        try:
            band.remove()
        except Exception:
            pass
    band = ax.fill_between(
        x_sel, lo_sel, hi_sel,
        alpha=0.18,
        color="#bbbbbb",
        linewidth=0
    )

    # Axes limits
    ax.set_xlim(x_sel[0], x_sel[-1])

    stack = np.vstack((yt_sel, yr_sel, lo_sel, hi_sel))
    ymin = np.nanmin(stack)
    ymax = np.nanmax(stack)
    pad = (ymax - ymin) * 0.06 if np.isfinite(ymax - ymin) else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)

    # Metrics
    rmse = (sse / cnt) ** 0.5 if cnt > 0 else float("nan")
    cov = (inside / cnt_cov) if cnt_cov > 0 else float("nan")

    ax.set_title(
        f"UNRATE forecasting (stream)\nRMSE RIDGE={rmse:.4f} · Coverage(95%)={cov:.3f}",
        color="white",
        fontsize=18
    )
    ph_text.markdown(
        f"<p style='color:white; font-size:22px; margin:0;'>"
        f"RMSE RIDGE = <b>{rmse:.4f}</b> &nbsp;|&nbsp; Coverage(95%) = <b>{cov:.3f}</b>"
        f"</p>",
        unsafe_allow_html=True
    )

    # Render JPEG -> st.image
    buf = BytesIO()
    fig.savefig(
        buf,
        format="jpg",
        dpi=DPI,
        bbox_inches="tight",
        pil_kwargs={"quality": JPEG_QUALITY, "optimize": True}
    )
    buf.seek(0)
    ph_fig.image(buf, use_container_width=True)

    # tiny sleep if ahead
    elapsed = time.time() - now
    if elapsed < REDRAW_EVERY:
        time.sleep(min(REDRAW_EVERY - elapsed, 0.002))