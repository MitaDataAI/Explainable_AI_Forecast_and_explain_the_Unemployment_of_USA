from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch


# ============================================================
# Utils
# ============================================================

def _fmt_df(df: pd.DataFrame) -> pd.DataFrame:
    """Arrondit proprement les floats pour affichage PDF."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return out


# ============================================================
# PDF builder
# ============================================================

def build_pdf(compare_dir: Path, *, title: str = "Comparison Report") -> Path:
    metrics_csv = compare_dir / "comparison_metrics.csv"
    error_csv = compare_dir / "error_analysis_mae_dm.csv"

    fig_dir = compare_dir / "figures"
    fig_forecasts = fig_dir / "global_forecasts.png"
    fig_errors = fig_dir / "global_errors.png"

    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing: {metrics_csv}")

    df_metrics = pd.read_csv(metrics_csv)

    df_error = None
    if error_csv.exists():
        df_error = pd.read_csv(error_csv)

    # Colonnes “préférées” (si dispo)
    preferred = [
        "method",
        "validation.rmse", "validation.mae", "validation.r2", "validation.n",
        "test.rmse", "test.mae", "test.r2", "test.n",
    ]
    cols = [c for c in preferred if c in df_metrics.columns]
    if not cols:
        cols = df_metrics.columns.tolist()

    df_metrics_small = _fmt_df(df_metrics[cols].copy())

    # Winner simple (min RMSE)
    def _winner(metric: str) -> str:
        if metric not in df_metrics.columns:
            return "NA"
        tmp = df_metrics.dropna(subset=[metric]).sort_values(metric, ascending=True)
        return str(tmp.iloc[0]["method"]) if len(tmp) else "NA"

    win_val = _winner("validation.rmse")
    win_test = _winner("test.rmse")

    out_pdf = compare_dir / "report.pdf"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
    ))
    styles.add(ParagraphStyle(
        name="Tiny",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#4b5563"),
    ))

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
        rightMargin=36, leftMargin=36, topMargin=32, bottomMargin=32,
    )

    story = []

    # ============================================================
    # Page 1 — Executive summary + metrics
    # ============================================================

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph(f"<b>Folder:</b> {compare_dir.as_posix()}", styles["Small"]))
    story.append(Spacer(1, 0.10 * inch))

    story.append(Paragraph("<b>Executive summary</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(f"• Best (validation RMSE): <b>{win_val}</b>", styles["Small"]))
    story.append(Paragraph(f"• Best (test RMSE): <b>{win_test}</b>", styles["Small"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Metrics (per run)", styles["Heading2"]))
    story.append(Spacer(1, 0.08 * inch))

    table_data = [df_metrics_small.columns.tolist()] + df_metrics_small.values.tolist()
    col_widths = [2.6 * inch] + [1.05 * inch] * (len(df_metrics_small.columns) - 1)

    tbl = Table(table_data, hAlign="LEFT", colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.HexColor("#e5e7eb")]),
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # ============================================================
    # Page 2 — Error analysis (MAE + DM)
    # ============================================================

    if df_error is not None and not df_error.empty:
        story.append(Paragraph("Error analysis (MAE + Diebold–Mariano)", styles["Heading1"]))
        story.append(Spacer(1, 0.12 * inch))

        df_error_small = df_error.fillna("")
        table_err = [df_error_small.columns.tolist()] + df_error_small.values.tolist()

        col_widths_err = [2.2 * inch] + [1.4 * inch] * (len(df_error_small.columns) - 1)

        tbl_err = Table(
            table_err,
            hAlign="LEFT",
            colWidths=col_widths_err,
            repeatRows=1,
        )
        tbl_err.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.whitesmoke, colors.HexColor("#e5e7eb")]),
        ]))
        story.append(tbl_err)
        story.append(PageBreak())

    # ============================================================
    # Page 3 — Figures
    # ============================================================

    story.append(Paragraph("Figures", styles["Heading1"]))
    story.append(Spacer(1, 0.12 * inch))

    def add_img(p: Path, caption: str):
        story.append(Paragraph(f"<b>{caption}</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.06 * inch))
        if p.exists():
            im = Image(str(p))
            im._restrictSize(10.0 * inch, 6.0 * inch)
            story.append(im)
        else:
            story.append(Paragraph(f"(Missing: {p.name})", styles["Small"]))
        story.append(Spacer(1, 0.20 * inch))

    add_img(fig_forecasts, "Global forecasts")
    add_img(fig_errors, "Absolute errors")

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