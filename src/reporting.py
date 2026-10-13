"""
reporting.py — CSV and PDF report export (Module F / Reports screen).
"""
from __future__ import annotations

import io
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(df: pd.DataFrame, include_index: bool = True) -> bytes:
    """Return a CSV byte-string from a DataFrame."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=include_index)
    return buffer.getvalue().encode("utf-8")


def export_forecast_csv(forecast_df: pd.DataFrame) -> bytes:
    """Export the 7-day forecast as CSV."""
    df = forecast_df.copy()
    df.index.name = "timestamp"
    df.insert(0, "data_source", "Synthetic — illustrative values only")
    return export_csv(df)


def export_plan_csv(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> bytes:
    """Export hourly and daily plan as a combined CSV."""
    buf = io.StringIO()
    buf.write("# SuryaSetu Weekly Plan Export\n")
    buf.write(f"# Generated: {datetime.now().isoformat()}\n")
    buf.write("# All values are SYNTHETIC and illustrative.\n\n")
    buf.write("## Hourly Plan\n")
    hourly_df.to_csv(buf)
    buf.write("\n## Daily Summary\n")
    daily_df.to_csv(buf)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

_DISCLAIMER = (
    "IMPORTANT: SuryaSetu is a decision-support prototype. All data in this "
    "report are SYNTHETIC and illustrative. They do not represent measurements "
    "from any real site. Results must not be used for engineering design, "
    "procurement, policy decisions or safety-critical purposes without "
    "independent expert validation."
)

_METHODOLOGY = (
    "Forecasting: Hourly generation and demand are predicted using (1) a "
    "baseline same-hour historical mean and (2) a scikit-learn RandomForest "
    "model trained on chronologically split data. "
    "Essential-service coverage: Critical loads (Primary Health Centre, "
    "Drinking-Water Facility) carry hard-constraint minimum energy requirements. "
    "Shortages are always reported explicitly. "
    "Optimisation: PuLP LP solver minimises unmet demand + 0.5 × grid usage "
    "subject to operating constraints. If infeasible, the conflict is exposed. "
    "All weather and demand inputs are generated from a deterministic synthetic "
    "model with plausible but unvalidated parameters."
)


def export_pdf(
    scenario_name: str,
    scenario_notes: str,
    balance_metrics: dict,
    daily_summary_df: Optional[pd.DataFrame],
    coverage_df: Optional[pd.DataFrame],
    comparison_df: Optional[pd.DataFrame],
    forecast_metrics_df: Optional[pd.DataFrame],
) -> bytes:
    """
    Generate a PDF summary report.

    Returns raw bytes (PDF).
    Falls back to a plain-text bytes object if ReportLab is unavailable.
    """
    if not REPORTLAB_AVAILABLE:
        return _fallback_text_report(
            scenario_name, scenario_notes, balance_metrics,
            daily_summary_df, coverage_df, comparison_df,
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"],
        fontSize=20, spaceAfter=6, textColor=colors.HexColor("#1a472a"),
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=13, spaceBefore=12, spaceAfter=4,
        textColor=colors.HexColor("#2d6a4f"),
    )
    body_style = styles["BodyText"]
    warn_style = ParagraphStyle(
        "Warn", parent=body_style,
        textColor=colors.HexColor("#b5390d"),
        spaceBefore=4,
    )

    story = []

    # Title
    story.append(Paragraph("SuryaSetu — Energy Resilience Report", title_style))
    story.append(Paragraph(
        f"<b>Community Energy Resilience Platform</b> | "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        body_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2d6a4f")))
    story.append(Spacer(1, 0.3 * cm))

    # Disclaimer
    story.append(Paragraph("⚠ Disclaimer", h2_style))
    story.append(Paragraph(_DISCLAIMER, warn_style))
    story.append(Spacer(1, 0.4 * cm))

    # Scenario
    story.append(Paragraph(f"Scenario: {scenario_name}", h2_style))
    story.append(Paragraph(scenario_notes or "No additional notes.", body_style))
    story.append(Spacer(1, 0.3 * cm))

    # Energy balance
    story.append(Paragraph("Weekly Energy Balance", h2_style))
    if balance_metrics:
        bm_rows = [["Metric", "Value"]] + [
            [k.replace("_", " ").title(), str(v)]
            for k, v in balance_metrics.items()
        ]
        story.append(_make_table(bm_rows))
    story.append(Spacer(1, 0.4 * cm))

    # Daily summary
    if daily_summary_df is not None and len(daily_summary_df) > 0:
        story.append(Paragraph("7-Day Daily Summary", h2_style))
        story.append(_df_to_table(daily_summary_df.round(2)))
        story.append(Spacer(1, 0.4 * cm))

    # Essential services
    if coverage_df is not None and len(coverage_df) > 0:
        story.append(Paragraph("Essential Service Coverage", h2_style))
        story.append(_df_to_table(coverage_df))
        story.append(Spacer(1, 0.4 * cm))

    # Schedule comparison
    if comparison_df is not None and len(comparison_df) > 0:
        story.append(Paragraph("Scheduling Method Comparison", h2_style))
        story.append(_df_to_table(comparison_df))
        story.append(Spacer(1, 0.4 * cm))

    # Forecast metrics
    if forecast_metrics_df is not None and len(forecast_metrics_df) > 0:
        story.append(Paragraph("Forecast Evaluation (Synthetic Data)", h2_style))
        story.append(_df_to_table(forecast_metrics_df))
        story.append(Spacer(1, 0.4 * cm))

    # Methodology
    story.append(PageBreak())
    story.append(Paragraph("Methodology & Limitations", h2_style))
    story.append(Paragraph(_METHODOLOGY, body_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Limitations: This prototype uses synthetic data only. It does not "
        "control physical equipment. Water-volume estimates use an illustrative "
        "pump-efficiency assumption and require expert hydraulic validation. "
        "Cost estimates are not included unless configured by the user. "
        "Real-world deployment requires expert review.",
        warn_style,
    ))

    doc.build(story)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_table(data: list[list], col_widths=None) -> Table:
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d6a4f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f7f4")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _df_to_table(df: pd.DataFrame) -> Table:
    df_disp = df.reset_index()
    header = list(df_disp.columns)
    rows = [header] + df_disp.astype(str).values.tolist()
    return _make_table(rows)


def _fallback_text_report(
    scenario_name: str,
    scenario_notes: str,
    balance_metrics: dict,
    daily_summary_df,
    coverage_df,
    comparison_df,
) -> bytes:
    """Plain-text fallback if ReportLab is unavailable."""
    lines = [
        "SuryaSetu — Energy Resilience Report",
        "=" * 50,
        f"Generated: {datetime.now().isoformat()}",
        "",
        "DISCLAIMER: All data are SYNTHETIC and illustrative.",
        "",
        f"Scenario: {scenario_name}",
        scenario_notes or "",
        "",
        "Energy Balance:",
    ]
    for k, v in balance_metrics.items():
        lines.append(f"  {k}: {v}")

    if daily_summary_df is not None:
        lines.append("\nDaily Summary:")
        lines.append(daily_summary_df.to_string())

    if coverage_df is not None:
        lines.append("\nService Coverage:")
        lines.append(coverage_df.to_string())

    if comparison_df is not None:
        lines.append("\nScheduling Comparison:")
        lines.append(comparison_df.to_string())

    return "\n".join(lines).encode("utf-8")
