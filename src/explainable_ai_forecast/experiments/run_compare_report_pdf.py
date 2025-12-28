from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# Helpers (format / tables)
# ============================================================

def _fmt_num(x, nd=4) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _fmt_df_numeric(df: pd.DataFrame, nd: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda v: _fmt_num(v, nd=nd))
    return out


def _styled_table(
    data: list,
    col_widths: list,
    *,
    header_bg: str = "#111827",
    header_text=colors.white,
    zebra: bool = True,
    right_align_from_col: int | None = None,
    repeat_rows: int = 1,
) -> Table:
    tbl = Table(data, hAlign="LEFT", colWidths=col_widths, repeatRows=repeat_rows)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), header_text),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if zebra and len(data) > 2:
        style_cmds.append(
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#e5e7eb")])
        )
    if right_align_from_col is not None:
        style_cmds.append(("ALIGN", (right_align_from_col, 1), (-1, -1), "RIGHT"))
        style_cmds.append(("ALIGN", (right_align_from_col, 0), (-1, 0), "CENTER"))
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


# ============================================================
# Statistical inference table (paper-like)
# ============================================================

def _load_stat_inference_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # common: index column exported as Unnamed: 0
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "feature"})
    if "feature" not in df.columns:
        df = df.rename(columns={df.columns[0]: "feature"})
    return df


def _build_si_grouped_table(df: pd.DataFrame) -> Tuple[list, List[str]]:
    """
    Expect flattened columns like:
      feature,
      "LINREG | βˢ", "LINREG | p-value", "LINREG | Γˢ", "LINREG | sig" (optional),
      "RIDGE | βˢ", ...
    Build a 2-row header table (paper style) grouped by model with 3 subcolumns: βˢ, p-value, Γˢ.
    If "sig" exists for a model, we append it to βˢ as stars.
    """
    pairs = []
    for c in df.columns:
        if c == "feature":
            continue
        if " | " in c:
            m, met = c.split(" | ", 1)
            pairs.append((m.strip(), met.strip(), c))

    if not pairs:
        raise ValueError(
            "Stat inference CSV columns not recognized. Expected 'MODEL | metric' columns."
        )

    model_order: List[str] = []
    by_model: dict[str, dict[str, str]] = {}
    for m, met, col in pairs:
        if m not in model_order:
            model_order.append(m)
            by_model[m] = {}
        by_model[m][met] = col

    # attach sig to βˢ if present
    df2 = df.copy()
    for m in model_order:
        if "βˢ" in by_model[m] and "sig" in by_model[m]:
            bcol = by_model[m]["βˢ"]
            scol = by_model[m]["sig"]
            df2[bcol] = df2[bcol].astype(str) + df2[scol].fillna("").astype(str)

    # headers
    header0 = ["Forecasting"]
    header1 = [""]

    for m in model_order:
        header0 += [m, "", ""]
        header1 += ["βˢ", "p-value", "Γˢ"]

    # rows
    rows = []
    for _, r in df2.iterrows():
        row = [str(r["feature"])]
        for m in model_order:
            # βˢ
            b = r.get(by_model[m].get("βˢ", ""), "")
            # p-value
            p = r.get(by_model[m].get("p-value", ""), "")
            # Γˢ
            g = r.get(by_model[m].get("Γˢ", ""), "")

            # small formatting if numeric-like
            row.append(str(b) if not pd.isna(b) else "")
            row.append(_fmt_num(p, nd=3) if str(p).strip() != "" else "")
            row.append(_fmt_num(g, nd=3) if str(g).strip() != "" else "")
        rows.append(row)

    return [header0, header1] + rows, model_order


def _paper_table_style(model_order: List[str]) -> TableStyle:
    """
    Build TableStyle for the 2-row header paper-like table.
    """
    ts = [
        ("FONTNAME", (0, 0), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 1), 9),
        ("FONTSIZE", (0, 2), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, 1), "CENTER"),
        ("ALIGN", (1, 2), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.black),
        ("LINEBELOW", (0, 1), (-1, 1), 1.0, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]

    # spans for each model name across its 3 columns in header row 0
    col = 1
    for _m in model_order:
        ts.append(("SPAN", (col, 0), (col + 2, 0)))
        # vertical separator after each model block (looks like paper)
        ts.append(("LINEAFTER", (col + 2, 0), (col + 2, -1), 0.6, colors.black))
        col += 3

    return TableStyle(ts)


# ============================================================
# PDF builder
# ============================================================

def build_pdf(compare_dir: Path, *, title: str = "Comparison Report") -> Path:
    # inputs
    metrics_csv = compare_dir / "comparison_metrics.csv"
    error_csv = compare_dir / "error_analysis_mae_dm.csv"

    # optional figures (if your pipeline creates them)
    fig_dir = compare_dir / "figures"
    fig_forecasts = fig_dir / "global_forecasts.png"
    fig_errors = fig_dir / "global_errors.png"

    # XAI artifacts
    xai_csv = compare_dir / "xai_perm_shap_all.csv"
    xai_img = compare_dir / "plots" / "importance_panels.png"

    # Statistical inference artifact (produced by run_statistical_inference.py)
    si_csv = compare_dir / "tables" / "statistical_inference_shapley_regression.csv"

    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing: {metrics_csv}")

    df_metrics = pd.read_csv(metrics_csv)

    df_error = None
    if error_csv.exists():
        df_error = pd.read_csv(error_csv)

    # pick "nice" columns if present
    preferred = [
        "method",
        "validation.rmse", "validation.mae", "validation.r2", "validation.n",
        "test.rmse", "test.mae", "test.r2", "test.n",
    ]
    cols = [c for c in preferred if c in df_metrics.columns]
    if not cols:
        cols = df_metrics.columns.tolist()

    df_metrics_small = _fmt_df_numeric(df_metrics[cols].copy(), nd=4)

    # winners
    def _winner(metric: str) -> str:
        if metric not in df_metrics.columns:
            return "NA"
        tmp = df_metrics.dropna(subset=[metric]).sort_values(metric, ascending=True)
        return str(tmp.iloc[0]["method"]) if len(tmp) else "NA"

    win_val = _winner("validation.rmse")
    win_test = _winner("test.rmse")

    out_pdf = compare_dir / "report.pdf"

    # styles
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=9,
            leading=11,
        )
    )

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(40, 25, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        canvas.drawRightString(doc.pagesize[0] - 40, 25, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=landscape(A4),
        rightMargin=36,
        leftMargin=36,
        topMargin=32,
        bottomMargin=32,
    )

    # IMPORTANT: story defined here
    story = []

    # ============================================================
    # Page 1 — Summary + metrics
    # ============================================================
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph(f"<b>Folder:</b> {compare_dir.as_posix()}", styles["Small"]))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("<b>Executive summary</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(f"• Best (validation RMSE): <b>{win_val}</b>", styles["Small"]))
    story.append(Paragraph(f"• Best (test RMSE): <b>{win_test}</b>", styles["Small"]))
    story.append(Spacer(1, 0.18 * inch))

    story.append(Paragraph("Metrics (per run)", styles["Heading2"]))
    story.append(Spacer(1, 0.08 * inch))

    table_data = [df_metrics_small.columns.tolist()] + df_metrics_small.values.tolist()
    col_widths = [2.6 * inch] + [1.05 * inch] * (len(df_metrics_small.columns) - 1)
    story.append(_styled_table(table_data, col_widths, header_bg="#111827", right_align_from_col=1))

    story.append(PageBreak())

    # ============================================================
    # Page 2 — Error analysis
    # ============================================================
    if df_error is not None and not df_error.empty:
        story.append(Paragraph("Error analysis (MAE + Diebold–Mariano)", styles["Heading1"]))
        story.append(Spacer(1, 0.12 * inch))

        df_error_small = df_error.fillna("")
        err_data = [df_error_small.columns.tolist()] + df_error_small.values.tolist()
        err_widths = [2.2 * inch] + [1.4 * inch] * (len(df_error_small.columns) - 1)
        story.append(_styled_table(err_data, err_widths, header_bg="#1f2937"))
        story.append(PageBreak())

    # ============================================================
    # Page 3 — XAI (table + panels image)
    # ============================================================
    story.append(Paragraph("Explainable AI (XAI)", styles["Heading1"]))
    story.append(Spacer(1, 0.12 * inch))

    # (A) XAI table from xai_perm_shap_all.csv
    if xai_csv.exists():
        df_xai = pd.read_csv(xai_csv)

        top_k = 15
        # choose sorting
        sort_col = "shapley_share" if "shapley_share" in df_xai.columns else "mean_perm_abs_error"
        df_xai = df_xai.sort_values(sort_col, ascending=False).head(top_k).copy()

        show_cols = ["model", "feature", "shapley_share", "mean_perm_abs_error", "mean_perm_deviance", "n_snapshots"]
        show_cols = [c for c in show_cols if c in df_xai.columns]
        df_show = df_xai[show_cols].copy()

        # formatting
        if "shapley_share" in df_show.columns:
            df_show["shapley_share"] = df_show["shapley_share"].map(lambda v: _fmt_num(v, nd=3))
        if "mean_perm_abs_error" in df_show.columns:
            df_show["mean_perm_abs_error"] = df_show["mean_perm_abs_error"].map(lambda v: _fmt_num(v, nd=4))
        if "mean_perm_deviance" in df_show.columns:
            df_show["mean_perm_deviance"] = df_show["mean_perm_deviance"].map(lambda v: _fmt_num(v, nd=4))
        if "n_snapshots" in df_show.columns:
            df_show["n_snapshots"] = df_show["n_snapshots"].map(lambda v: "" if pd.isna(v) else str(int(float(v))))

        story.append(Paragraph("Top features (SHAP share + permutation importance)", styles["Heading2"]))
        story.append(Spacer(1, 0.06 * inch))

        xai_data = [df_show.columns.tolist()] + df_show.values.tolist()

        # widths for A4 landscape
        widths = []
        for c in df_show.columns:
            if c == "model":
                widths.append(1.0 * inch)
            elif c == "feature":
                widths.append(2.2 * inch)
            else:
                widths.append(1.25 * inch)

        story.append(_styled_table(xai_data, widths, header_bg="#111827", right_align_from_col=2))
    else:
        story.append(Paragraph(f"(Missing: {xai_csv.name})", styles["Small"]))

    story.append(Spacer(1, 0.18 * inch))

    # (B) panels image
    story.append(Paragraph("Variable importance panels", styles["Heading2"]))
    story.append(Spacer(1, 0.06 * inch))
    if xai_img.exists():
        im = Image(str(xai_img))
        im._restrictSize(10.0 * inch, 3.2 * inch)
        story.append(im)
    else:
        story.append(Paragraph(f"(Missing: {xai_img.as_posix()})", styles["Small"]))

    story.append(PageBreak())

    # ============================================================
    # Page 4 — Statistical inference (paper-like table)
    # ============================================================
    story.append(Paragraph("Statistical inference (Shapley regressions)", styles["Heading1"]))
    story.append(Spacer(1, 0.12 * inch))

    if si_csv.exists():
        df_si = _load_stat_inference_csv(si_csv)

        # Build grouped header table
        table_si, model_order = _build_si_grouped_table(df_si)

        # widths: first col wider, then 3 cols per model
        col_widths_si = [2.6 * inch] + [0.95 * inch] * (3 * len(model_order))

        tbl_si = Table(table_si, hAlign="LEFT", colWidths=col_widths_si, repeatRows=2)
        tbl_si.setStyle(_paper_table_style(model_order))
        story.append(tbl_si)
    else:
        story.append(Paragraph(f"(Missing: {si_csv.as_posix()})", styles["Small"]))

    story.append(PageBreak())

    # ============================================================
    # Page 5 — Figures (optional)
    # ============================================================
    story.append(Paragraph("Figures", styles["Heading1"]))
    story.append(Spacer(1, 0.12 * inch))

    def _add_img(p: Path, caption: str):
        story.append(Paragraph(f"<b>{caption}</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.06 * inch))
        if p.exists():
            im = Image(str(p))
            im._restrictSize(10.0 * inch, 6.0 * inch)
            story.append(im)
        else:
            story.append(Paragraph(f"(Missing: {p.as_posix()})", styles["Small"]))
        story.append(Spacer(1, 0.20 * inch))

    _add_img(fig_forecasts, "Global forecasts")
    _add_img(fig_errors, "Absolute errors")

    # build
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return out_pdf


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--compare-dir", type=str, required=True)
    p.add_argument("--title", type=str, default="Comparison Report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = build_pdf(Path(args.compare_dir), title=args.title)
    print(f"[OK] PDF saved to: {out}")


if __name__ == "__main__":
    main()