"""
pdf_report.py - Clinical PDF report generation for MyoControl Suite

Produces a single-file, printable clinical/technical report from an
EMGEngine result dict: acquisition parameters, signal quality, clinical
interpretation, per-channel summary, feature statistics, and a signal
plot. Intended as documentation you can hand to a clinician or file
alongside a research dataset — not a substitute for a diagnosis.
"""
import io
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless, no display backend required
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable,
)

ACCENT = colors.HexColor("#2F5FE0")
TEXT_SOFT = colors.HexColor("#5B6472")
BORDER = colors.HexColor("#E3E8EF")


def _signal_plot_image(raw_signal: np.ndarray, filtered_signal: np.ndarray,
                        sampling_rate: int, channel: int = 0) -> io.BytesIO:
    """Render raw vs filtered signal as a PNG for embedding in the PDF."""
    t = np.arange(len(raw_signal)) / sampling_rate
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.2), sharex=True, dpi=150)
    axes[0].plot(t, raw_signal[:, channel], color="#94A3B8", linewidth=0.5)
    axes[0].set_title("Raw EMG", fontsize=9, loc="left", color="#334155")
    axes[0].tick_params(labelsize=7)
    axes[1].plot(t, filtered_signal[:, channel], color="#2F5FE0", linewidth=0.6)
    axes[1].set_title("Filtered EMG", fontsize=9, loc="left", color="#334155")
    axes[1].set_xlabel("Time (s)", fontsize=8)
    axes[1].tick_params(labelsize=7)
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(
    result: Dict[str, Any],
    raw_signal: Optional[np.ndarray] = None,
    filtered_signal: Optional[np.ndarray] = None,
    acquisition_info: Optional[Dict[str, Any]] = None,
    classifier_metrics: Optional[Dict[str, Any]] = None,
    author: str = "",
) -> bytes:
    """Build the report and return it as PDF bytes (write with open(path,'wb'))."""
    acquisition_info = acquisition_info or {}
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=20 * mm, bottomMargin=18 * mm, leftMargin=20 * mm, rightMargin=20 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontSize=20, leading=24, fontName="Helvetica-Bold", textColor=colors.HexColor("#0F172A")))
    styles.add(ParagraphStyle(name="ReportSubtitle", fontSize=10.5, textColor=TEXT_SOFT, spaceAfter=2))
    styles.add(ParagraphStyle(name="SectionHeading", fontSize=13, fontName="Helvetica-Bold", textColor=colors.HexColor("#0F172A"), spaceBefore=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodySmall", fontSize=9.5, leading=13, textColor=colors.HexColor("#334155")))
    styles.add(ParagraphStyle(name="Disclaimer", fontSize=8, leading=11, textColor=TEXT_SOFT))

    story = []

    # ---- Header ----
    story.append(Paragraph("MyoControl Suite &mdash; Analysis Report", styles["ReportTitle"]))
    sub = f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    if author:
        sub += f" &nbsp;&middot;&nbsp; Built by {author}"
    story.append(Paragraph(sub, styles["ReportSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceBefore=6, spaceAfter=12))

    # ---- Acquisition summary table ----
    meta = result['metadata']
    sq = result['signal_quality']
    ci = result['clinical_interpretation']

    acq_rows = [
        ["Source", acquisition_info.get('source', 'N/A')],
        ["Contraction type", str(acquisition_info.get('gesture') or 'N/A')],
        ["Contraction intensity", f"{acquisition_info['intensity']:.2f}x" if acquisition_info.get('intensity') is not None else 'N/A'],
        ["Sampling rate", f"{meta['sampling_rate']} Hz"],
        ["Duration", f"{meta['duration_seconds']:.2f} s ({meta['n_samples']} samples)"],
        ["Channels", str(meta['n_channels'])],
        ["Filter band", f"{meta['filter_config'].get('cutoff_low', 20):.0f}\u2013{meta['filter_config'].get('cutoff_high', 450):.0f} Hz, order {meta['filter_config'].get('filter_order', 4)}"],
    ]
    story.append(Paragraph("Acquisition", styles["SectionHeading"]))
    story.append(_make_table(acq_rows))

    # ---- Signal quality / clinical interpretation ----
    quality_rows = [
        ["Signal quality", f"{sq['snr_quality'].title()} (SNR {sq['mean_snr_db']:.1f} dB)"],
        ["Artifact detected", f"{'Yes' if sq['artifact_detected'] else 'No'} ({sq['artifact_percentage']:.2f}% of samples)"],
        ["Muscle activity", ci['muscle_activity'].replace('_', ' ').title()],
        ["Activation pattern", ci['activation_pattern'].replace('_', ' ').title()],
    ]
    story.append(Paragraph("Signal Quality &amp; Clinical Interpretation", styles["SectionHeading"]))
    story.append(_make_table(quality_rows))

    # ---- Signal plot ----
    if raw_signal is not None and filtered_signal is not None:
        story.append(Paragraph("Signal", styles["SectionHeading"]))
        img_buf = _signal_plot_image(raw_signal, filtered_signal, meta['sampling_rate'])
        story.append(Image(img_buf, width=170 * mm, height=85 * mm))

    # ---- Per-channel summary ----
    story.append(Paragraph("Per-Channel Summary", styles["SectionHeading"]))
    summary = result['summary_statistics']
    if summary:
        first_ch = next(iter(summary.values()))
        cols = list(first_ch.keys())
        header = ["channel"] + cols
        rows = [header]
        for ch_key, ch_summary in summary.items():
            row = [ch_key] + [_fmt(ch_summary.get(c)) for c in cols]
            rows.append(row)
        story.append(_make_table(rows, header_row=True))

    # ---- Classifier metrics (optional) ----
    if classifier_metrics:
        story.append(Paragraph("Gesture Classifier", styles["SectionHeading"]))
        rows = [
            ["Model", str(classifier_metrics.get('model_type', 'N/A'))],
            ["Training samples", str(classifier_metrics.get('n_samples', 'N/A'))],
            ["Train accuracy", f"{classifier_metrics.get('train_accuracy', 0) * 100:.1f}%"],
        ]
        if classifier_metrics.get('loso_accuracy_mean') is not None:
            ci95 = classifier_metrics.get('loso_accuracy_ci95')
            ci_str = f" (95% CI {ci95[0]*100:.1f}\u2013{ci95[1]*100:.1f}%)" if ci95 else ""
            rows.append(["LOSO accuracy (cross-subject)",
                         f"{classifier_metrics['loso_accuracy_mean'] * 100:.1f}%{ci_str}"])
            rows.append(["LOSO folds (subjects)", str(classifier_metrics.get('loso_n_folds', 'N/A'))])
        story.append(_make_table(rows))

    # ---- Disclaimer ----
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6))
    story.append(Paragraph(
        "This report is generated by MyoControl Suite for research and technical "
        "documentation purposes. Signal quality labels and activity classifications "
        "are heuristic and have not been calibrated against expert-rated clinical "
        "recordings; the classifier, unless trained on your own labeled data, is "
        "demonstrating pattern-recognition capability on synthetic data only. This "
        "report is not a medical diagnosis and should not be used as one.",
        styles["Disclaimer"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _make_table(rows, header_row: bool = False) -> Table:
    t = Table(rows, hAlign="LEFT")
    style = [
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#334155")),
        ('LINEBELOW', (0, 0), (-1, -1), 0.4, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]
    if header_row:
        style += [
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ]
    else:
        style += [('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold')]
    t.setStyle(TableStyle(style))
    return t
