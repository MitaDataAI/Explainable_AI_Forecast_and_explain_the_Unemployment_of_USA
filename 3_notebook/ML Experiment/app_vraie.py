"""
PySide6 + PyQtGraph — UNRATE + PI95 + C(t) + live metrics
✅ Interactif:
- Hover: tooltip (date, y_true, y_ridge, PI95, C)
- Clic gauche: freeze / unfreeze du curseur
- Clic droit: unfreeze
- Sélection de période (drag): LinearRegionItem + RMSE/Coverage sur la sélection
- Raccourcis: T (toggle true), P (toggle pred), I (toggle interval)

pip install pyside6 pyqtgraph pandas numpy pyarrow python-dateutil

Run:
    python app_unrate_stream_qt_interactive.py

Expected files:
    outputs/streamlit/bkt_ridge_final.parquet
    outputs/streamlit/meta_ridge.json   (or meta_ridge.pkl)
"""

import sys
import json
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


# =========================
# Paths
# =========================
DATA_DIR = Path("outputs/streamlit")
BKT_PATH = DATA_DIR / "bkt_ridge_final.parquet"
META_JSON = DATA_DIR / "meta_ridge.json"
META_PKL  = DATA_DIR / "meta_ridge.pkl"


# =========================
# Helpers: find PI cols
# =========================
def _find_pi_cols(df: pd.DataFrame, model_col: str) -> tuple[str, str]:
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
        f"PI 95% introuvable pour {model_col}. "
        f"Attendu: '{model_col}-lo-95' / '{model_col}-hi-95' (ou équivalent)."
    )


def load_bkt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")

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


def load_meta() -> dict:
    if META_JSON.exists():
        with open(META_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    if META_PKL.exists():
        with open(META_PKL, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"meta_ridge not found: {META_JSON} or {META_PKL}")


def build_c_hist(meta_ridge: dict) -> pd.DataFrame:
    if "alpha_history" not in meta_ridge:
        raise KeyError("meta_ridge must contain 'alpha_history'.")

    h = pd.DataFrame(meta_ridge["alpha_history"]).copy()
    needed = {"cutoff_start", "n_windows", "alpha"}
    miss = needed - set(h.columns)
    if miss:
        raise KeyError(f"alpha_history incomplete: {sorted(miss)}")

    h["cutoff_start"] = pd.to_datetime(h["cutoff_start"])
    h["cutoff_end"] = h.apply(
        lambda r: r["cutoff_start"] + relativedelta(months=int(r["n_windows"]) - 1),
        axis=1
    )
    h = h.sort_values("cutoff_start").reset_index(drop=True)
    h["C"] = pd.to_numeric(h["alpha"], errors="coerce")
    return h[["cutoff_start", "cutoff_end", "C"]]


def smooth_ema_arr(arr: np.ndarray, span: int) -> np.ndarray:
    s = pd.Series(arr)
    if span and span > 1:
        return s.ewm(span=span, adjust=False).mean().astype(np.float32).values
    return s.astype(np.float32).values


def to_seconds(dt_index: pd.DatetimeIndex) -> np.ndarray:
    # pyqtgraph.DateAxisItem expects Unix time (seconds)
    return (dt_index.view("int64") / 1e9).astype(np.float64)


# =========================
# Streaming config
# =========================
@dataclass
class StreamCfg:
    fps: int = 60
    window: int = 600
    max_pts: int = 1200
    smooth_span: int = 1


# =========================
# Main App
# =========================
class UnrateStreamer(QtWidgets.QMainWindow):
    def __init__(self, df: pd.DataFrame, c_hist: pd.DataFrame, cfg: StreamCfg):
        super().__init__()
        self.setWindowTitle("UNRATE — RIDGE + Conformal + C(t) (Interactive)")
        self.resize(1400, 880)

        self.cfg = cfg

        # ---- Prepare arrays ----
        idx = pd.DatetimeIndex(df["ds"])
        self.t = idx
        self.x_all = to_seconds(idx)

        y_true  = df["y_true"].to_numpy(dtype=np.float32)
        y_ridge = df["y_ridge"].to_numpy(dtype=np.float32)
        y_lo    = df["y_lo"].to_numpy(dtype=np.float32)
        y_hi    = df["y_hi"].to_numpy(dtype=np.float32)

        self.y_true_all  = smooth_ema_arr(y_true,  cfg.smooth_span)
        self.y_ridge_all = smooth_ema_arr(y_ridge, cfg.smooth_span)
        self.y_lo_all    = smooth_ema_arr(y_lo,    cfg.smooth_span)
        self.y_hi_all    = smooth_ema_arr(y_hi,    cfg.smooth_span)

        # C(t) step function via merge_asof
        tmp_ds = pd.DataFrame({"ds": idx})
        tmp_c  = c_hist.sort_values("cutoff_start").rename(columns={"cutoff_start": "ds"})
        tmp_c  = tmp_c[["ds", "cutoff_end", "C"]]

        c_map = pd.merge_asof(
            tmp_ds.sort_values("ds"),
            tmp_c.sort_values("ds"),
            on="ds",
            direction="backward"
        )
        self.C_all = c_map["C"].to_numpy(dtype=np.float32)

        self.N = len(self.x_all)
        self.k = 0

        # ---- metrics accumulators ----
        self.sse = 0.0
        self.cnt = 0
        self.inside = 0
        self.cnt_cov = 0

        self.valid_true = np.isfinite(self.y_true_all)
        self.valid_ridge = np.isfinite(self.y_ridge_all)

        # ---- UI ----
        self._build_ui()

        # ---- Interactivité souris ----
        self._hover_enabled = True
        self._frozen = False
        self._last_mouse_x = None

        self._proxy_move = pg.SignalProxy(
            self.p1.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved
        )
        self._proxy_click = pg.SignalProxy(
            self.p1.scene().sigMouseClicked,
            rateLimit=60,
            slot=self._on_mouse_clicked
        )

        self.tip = pg.TextItem("", anchor=(0, 1))
        self.tip.setZValue(10)
        self.p1.addItem(self.tip)

        # selection metrics
        self.region.sigRegionChanged.connect(self._update_selection_metrics)
        self._update_selection_metrics()

        # ---- Timer ----
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000 / max(1, cfg.fps)))

    def _build_ui(self):
        # Dark theme
        pg.setConfigOption("background", (0, 0, 0))
        pg.setConfigOption("foreground", (230, 230, 230))
        pg.setConfigOptions(antialias=True)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Top metrics
        self.lbl = QtWidgets.QLabel("RMSE=... | Coverage(95%)=... | C=...")
        self.lbl.setStyleSheet("color: white; font-size: 18px;")
        layout.addWidget(self.lbl)

        # Selection metrics
        self.lbl_sel = QtWidgets.QLabel("Selection RMSE=... | Coverage=...")
        self.lbl_sel.setStyleSheet("color: white; font-size: 14px;")
        layout.addWidget(self.lbl_sel)

        # Plot area
        self.gw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.gw, 1)

        # ---- Plot 1: UNRATE ----
        date_axis_1 = pg.DateAxisItem(orientation="bottom")
        date_axis_1.setStyle(tickTextOffset=10)

        self.p1 = self.gw.addPlot(row=0, col=0, axisItems={"bottom": date_axis_1})
        self.p1.showGrid(x=True, y=True, alpha=0.25)
        self.p1.setLabel("left", "UNRATE")
        self.p1.setLabel("bottom", "Date")
        self.p1.addLegend(offset=(10, 10))

        self.curve_true = self.p1.plot(
            pen=pg.mkPen(color=(135, 206, 235), width=2),
            name="UNRATE (true)"
        )
        self.curve_ridge = self.p1.plot(
            pen=pg.mkPen(color=(255, 165, 0), width=2, style=QtCore.Qt.DashLine),
            name="RIDGE (pred)"
        )

        self.curve_lo = self.p1.plot(
            pen=pg.mkPen(color=(170, 170, 170), width=1, style=QtCore.Qt.DotLine),
            name="PI95 lo"
        )
        self.curve_hi = self.p1.plot(
            pen=pg.mkPen(color=(170, 170, 170), width=1, style=QtCore.Qt.DotLine),
            name="PI95 hi"
        )
        self.band = pg.FillBetweenItem(
            self.curve_lo,
            self.curve_hi,
            brush=pg.mkBrush(170, 170, 170, 50)
        )
        self.p1.addItem(self.band)

        # Crosshair
        self.vline1 = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen((120, 120, 120), width=1, style=QtCore.Qt.DashLine)
        )
        self.p1.addItem(self.vline1)

        # Selection region (drag)
        if self.N > 2:
            init_right = self.x_all[min(self.N - 1, 120)]
        else:
            init_right = self.x_all[-1] if self.N else 0
        self.region = pg.LinearRegionItem(
            values=[self.x_all[0] if self.N else 0, init_right],
            movable=True
        )
        self.region.setZValue(9)
        self.p1.addItem(self.region)

        # Mouse interactions
        self.p1.setMouseEnabled(x=True, y=True)
        self.p1.getViewBox().setMouseMode(pg.ViewBox.RectMode)

        # ---- Plot 2: C(t) ----
        date_axis_2 = pg.DateAxisItem(orientation="bottom")
        date_axis_2.setStyle(tickTextOffset=10)

        self.p2 = self.gw.addPlot(row=1, col=0, axisItems={"bottom": date_axis_2})
        self.p2.showGrid(x=True, y=True, alpha=0.25)
        self.p2.setLabel("left", "C")
        self.p2.setLabel("bottom", "Date")
        self.p2.addLegend(offset=(10, 10))

        self.curve_c = self.p2.plot(
            pen=pg.mkPen(color=(0, 255, 153), width=2),
            name="C(t)"
        )

        self.vline2 = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen((120, 120, 120), width=1, style=QtCore.Qt.DashLine)
        )
        self.p2.addItem(self.vline2)

        # Link x-axes (zoom/pan sync)
        self.p2.setXLink(self.p1)

        self.p2.setMouseEnabled(x=True, y=True)
        self.p2.getViewBox().setMouseMode(pg.ViewBox.RectMode)

        # Controls row
        ctrl = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl)

        self.btn_play = QtWidgets.QPushButton("Pause")
        self.btn_play.clicked.connect(self._toggle_play)
        ctrl.addWidget(self.btn_play)

        self.sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld.setRange(0, max(0, self.N - 1))
        self.sld.valueChanged.connect(self._seek)
        ctrl.addWidget(self.sld, 1)

        self.lbl_pos = QtWidgets.QLabel("")
        self.lbl_pos.setStyleSheet("color: white; font-size: 14px;")
        ctrl.addWidget(self.lbl_pos)

        # Shortcuts
        self.shortcut_t = QtGui.QShortcut(QtGui.QKeySequence("T"), self)
        self.shortcut_t.activated.connect(lambda: self._toggle_item(self.curve_true))

        self.shortcut_p = QtGui.QShortcut(QtGui.QKeySequence("P"), self)
        self.shortcut_p.activated.connect(lambda: self._toggle_item(self.curve_ridge))

        self.shortcut_i = QtGui.QShortcut(QtGui.QKeySequence("I"), self)
        self.shortcut_i.activated.connect(self._toggle_interval)

        self._playing = True

    # ---------- Interactions ----------
    def _nearest_index(self, x_sec: float) -> int:
        i = int(np.searchsorted(self.x_all, x_sec))
        i = max(0, min(i, self.N - 1))
        if 0 < i < self.N:
            if abs(self.x_all[i] - x_sec) > abs(self.x_all[i - 1] - x_sec):
                i -= 1
        return i

    def _format_tip(self, i: int) -> str:
        dt = self.t[i].strftime("%Y-%m")
        yt = float(self.y_true_all[i]) if np.isfinite(self.y_true_all[i]) else np.nan
        yr = float(self.y_ridge_all[i]) if np.isfinite(self.y_ridge_all[i]) else np.nan
        lo = float(self.y_lo_all[i]) if np.isfinite(self.y_lo_all[i]) else np.nan
        hi = float(self.y_hi_all[i]) if np.isfinite(self.y_hi_all[i]) else np.nan
        c  = float(self.C_all[i]) if np.isfinite(self.C_all[i]) else np.nan

        def f(v):
            return "NaN" if not np.isfinite(v) else f"{v:.3f}"

        return (
            f"<b>{dt}</b><br>"
            f"y_true: {f(yt)}<br>"
            f"y_ridge: {f(yr)}<br>"
            f"PI95: [{f(lo)}, {f(hi)}]<br>"
            f"C(t): {('NaN' if not np.isfinite(c) else f'{c:.4g}')}"
        )

    def _on_mouse_moved(self, evt):
        if (not self._hover_enabled) or self._frozen or self.N == 0:
            return

        pos = evt[0]
        if not self.p1.sceneBoundingRect().contains(pos):
            return

        vb = self.p1.getViewBox()
        mousePoint = vb.mapSceneToView(pos)
        x = float(mousePoint.x())
        self._last_mouse_x = x

        i = self._nearest_index(x)
        x_i = float(self.x_all[i])

        self.vline1.setPos(x_i)
        self.vline2.setPos(x_i)

        y_ref = self.y_ridge_all[i] if np.isfinite(self.y_ridge_all[i]) else self.y_true_all[i]
        if not np.isfinite(y_ref):
            y_ref = float(mousePoint.y())

        self.tip.setHtml(self._format_tip(i))
        self.tip.setPos(x_i, float(y_ref))

    def _on_mouse_clicked(self, evt):
        if self.N == 0:
            return

        click = evt[0]
        if click.button() == QtCore.Qt.LeftButton:
            self._frozen = not self._frozen
            if self._frozen and self._last_mouse_x is not None:
                i = self._nearest_index(self._last_mouse_x)
                x_i = float(self.x_all[i])
                self.vline1.setPos(x_i)
                self.vline2.setPos(x_i)
                y_ref = self.y_ridge_all[i] if np.isfinite(self.y_ridge_all[i]) else self.y_true_all[i]
                if not np.isfinite(y_ref):
                    y_ref = 0.0
                self.tip.setHtml(self._format_tip(i))
                self.tip.setPos(x_i, float(y_ref))

        elif click.button() == QtCore.Qt.RightButton:
            self._frozen = False

    # ---------- Controls ----------
    def _toggle_item(self, item):
        item.setVisible(not item.isVisible())
        self.band.setVisible(self.curve_lo.isVisible() and self.curve_hi.isVisible())

    def _toggle_interval(self):
        vis = not self.curve_lo.isVisible()
        self.curve_lo.setVisible(vis)
        self.curve_hi.setVisible(vis)
        self.band.setVisible(vis)

    def _toggle_play(self):
        self._playing = not self._playing
        self.btn_play.setText("Pause" if self._playing else "Play")

    def _seek(self, value: int):
        self.k = int(value)
        self._render()

    # ---------- Streaming ----------
    def _tick(self):
        if not self._playing:
            return
        if self.k >= self.N:
            self._playing = False
            self.btn_play.setText("Play")
            return

        k = self.k

        # RMSE incremental
        if self.valid_true[k] and self.valid_ridge[k]:
            diff = float(self.y_true_all[k] - self.y_ridge_all[k])
            self.sse += diff * diff
            self.cnt += 1

        # Coverage incremental
        yt = self.y_true_all[k]
        lo = self.y_lo_all[k]
        hi = self.y_hi_all[k]
        if np.isfinite(yt) and np.isfinite(lo) and np.isfinite(hi):
            self.cnt_cov += 1
            if lo <= yt <= hi:
                self.inside += 1

        self.k += 1

        self.sld.blockSignals(True)
        self.sld.setValue(self.k - 1)
        self.sld.blockSignals(False)

        self._render()

    def _downsample(self, x: np.ndarray, *ys: np.ndarray):
        n = len(x)
        if n <= self.cfg.max_pts:
            return (x,) + ys

        step = max(1, n // self.cfg.max_pts)
        x_sel = x[::step]
        ys_sel = tuple(y[::step] for y in ys)

        if x_sel[-1] != x[-1]:
            x_sel = np.concatenate([x_sel, x[-1:]])
            ys_sel = tuple(
                np.concatenate([y, y_full[-1:]])
                for y, y_full in zip(ys_sel, ys)
            )
        return (x_sel,) + ys_sel

    def _render(self):
        if self.N == 0:
            return

        stop = max(1, min(self.k, self.N))
        start = max(0, stop - self.cfg.window)

        x = self.x_all[start:stop]
        yt = self.y_true_all[start:stop]
        yr = self.y_ridge_all[start:stop]
        lo = self.y_lo_all[start:stop]
        hi = self.y_hi_all[start:stop]
        c  = self.C_all[start:stop]

        x, yt, yr, lo, hi, c = self._downsample(x, yt, yr, lo, hi, c)

        self.curve_true.setData(x, yt)
        self.curve_ridge.setData(x, yr)
        self.curve_lo.setData(x, lo)
        self.curve_hi.setData(x, hi)
        self.curve_c.setData(x, c)

        x_now = float(self.x_all[max(0, stop - 1)])
        if not self._frozen:
            self.vline1.setPos(x_now)
            self.vline2.setPos(x_now)

        # y ranges
        stack = np.vstack([yt, yr, lo, hi])
        if np.any(np.isfinite(stack)):
            ymin = float(np.nanmin(stack))
            ymax = float(np.nanmax(stack))
            pad = (ymax - ymin) * 0.06 if np.isfinite(ymax - ymin) else 1.0
            self.p1.setYRange(ymin - pad, ymax + pad, padding=0)

        if np.any(np.isfinite(c)):
            cmin = float(np.nanmin(c))
            cmax = float(np.nanmax(c))
            cpad = (cmax - cmin) * 0.10 if np.isfinite(cmax - cmin) else 1.0
            self.p2.setYRange(cmin - cpad, cmax + cpad, padding=0)

        # Metrics
        rmse = (self.sse / self.cnt) ** 0.5 if self.cnt > 0 else float("nan")
        cov = (self.inside / self.cnt_cov) if self.cnt_cov > 0 else float("nan")
        c_now = float(self.C_all[max(0, stop - 1)]) if np.isfinite(self.C_all[max(0, stop - 1)]) else float("nan")

        dt_str = self.t[max(0, stop - 1)].strftime("%Y-%m") if stop > 0 else ""
        self.lbl.setText(f"RMSE = {rmse:.4f}   |   Coverage(95%) = {cov:.3f}   |   C = {c_now:.4g}")
        self.lbl_pos.setText(f"{dt_str}   ({stop}/{self.N})")

        # Keep x-range to window
        if len(x) >= 2:
            self.p1.setXRange(float(x[0]), float(x[-1]), padding=0)

    def _update_selection_metrics(self):
        if self.N == 0:
            self.lbl_sel.setText("Selection: (vide)")
            return

        x0, x1 = self.region.getRegion()
        left = float(min(x0, x1))
        right = float(max(x0, x1))

        i0 = self._nearest_index(left)
        i1 = self._nearest_index(right)
        if i1 <= i0:
            self.lbl_sel.setText("Selection: (vide)")
            return

        yt = self.y_true_all[i0:i1 + 1]
        yr = self.y_ridge_all[i0:i1 + 1]
        lo = self.y_lo_all[i0:i1 + 1]
        hi = self.y_hi_all[i0:i1 + 1]

        mask_err = np.isfinite(yt) & np.isfinite(yr)
        rmse = float(np.sqrt(np.mean((yt[mask_err] - yr[mask_err]) ** 2))) if np.any(mask_err) else float("nan")

        mask_cov = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
        cov = float(np.mean((yt[mask_cov] >= lo[mask_cov]) & (yt[mask_cov] <= hi[mask_cov]))) if np.any(mask_cov) else float("nan")

        d0 = self.t[i0].strftime("%Y-%m")
        d1 = self.t[i1].strftime("%Y-%m")
        self.lbl_sel.setText(f"Selection {d0} → {d1} | RMSE = {rmse:.4f} | Coverage(95%) = {cov:.3f}")

    def showEvent(self, event):
        super().showEvent(event)
        self._render()


def main():
    df = load_bkt(BKT_PATH)
    meta = load_meta()
    c_hist = build_c_hist(meta)

    cfg = StreamCfg(
        fps=60,
        window=600,
        max_pts=1200,
        smooth_span=1
    )

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("UNRATE Interactive Stream")

    f = QtGui.QFont("Segoe UI", 10)
    app.setFont(f)

    w = UnrateStreamer(df, c_hist, cfg)
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()