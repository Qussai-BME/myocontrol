"""
app.py - MyoControl Suite Streamlit UI

Design language: clean clinical-professional dashboard. Colors follow
Streamlit's own System/Light/Dark theme picker (top-right menu) via a
small client-side sync script -- see THEME_SYNC_SCRIPT below.

6 tabs: Signal Analysis | Features | Spectral | Classification | Report | Statistics
"""
import sys
import os
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time

from src.core.config import EMGConfig
from src.core.engine import EMGEngine
from src.core.classifier import (
    EMGClassifier, generate_synthetic_emg_dataset,
)
from src.core.simulator import EMGSimulator
from src.core.features import (
    TIME_FEATURES, FREQ_FEATURES,
    extract_window_features,
)
from src.core import database as db
from src.core.pdf_report import generate_pdf_report
from src.core.dataset_loader import (
    load_features_csv, robust_read_table, load_raw_labeled_table,
    load_ninapro_mat, inspect_mat_fields, load_mat_numeric_field,
    concatenate_subjects,
)

try:
    import shap  # noqa: F401
    SHAP_IMPORTABLE = True
except ImportError:
    SHAP_IMPORTABLE = False

# ==========================================================
# Page config
# ==========================================================
st.set_page_config(
    page_title="MyoControl Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

AUTHOR = "Qussai Adlbi"

# ==========================================================
# Theme sync — Streamlit's own System/Light/Dark picker (top-right menu)
# does not expose its choice as a CSS variable or DOM attribute, so plain
# `prefers-color-scheme` CSS can drift out of sync with what the user
# actually selected. This tiny script reads the *actual* rendered
# background Streamlit is using (from the parent document, since
# components.html runs in an iframe) and mirrors it onto
# <html data-app-theme="light|dark">, which our CSS below keys off.
# ==========================================================
THEME_SYNC_SCRIPT = """
<script>
(function() {
    function luminance(rgbStr) {
        const m = rgbStr.match(/\\d+/g);
        if (!m) return 255;
        const [r, g, b] = m.map(Number);
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
    function sync() {
        try {
            const doc = window.parent.document;
            const bg = getComputedStyle(doc.body).backgroundColor;
            const isDark = luminance(bg) < 128;
            doc.documentElement.setAttribute('data-app-theme', isDark ? 'dark' : 'light');
        } catch (e) {}
    }
    sync();
    setInterval(sync, 500);
})();
</script>
"""
components.html(THEME_SYNC_SCRIPT, height=0)

# ==========================================================
# Global CSS — tokens default to light, overridden by the attribute the
# script above sets once it reads Streamlit's actual active theme.
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #FFFFFF;
    --bg-panel: #F8FAFC;
    --bg-alt: #EFF3F8;
    --text: #0F172A;
    --text-soft: #5B6472;
    --border: #E3E8EF;
    --accent: #2F5FE0;
    --accent-hover: #2450C4;
    --accent-soft: rgba(47,95,224,0.10);
    --accent-2: #F59E0B;
    --success: #0FA968;
    --success-soft: rgba(15,169,104,0.12);
    --warn-soft: rgba(245,158,11,0.14);
    --danger: #E4483A;
    --danger-soft: rgba(228,72,58,0.12);
    --shadow: 0 1px 2px rgba(15,23,42,0.05), 0 4px 14px rgba(15,23,42,0.04);
}
html[data-app-theme="dark"] {
    --bg: #0C1017;
    --bg-panel: #131A26;
    --bg-alt: #1A2331;
    --text: #E8ECF2;
    --text-soft: #94A0B4;
    --border: #232E40;
    --accent: #5B8CFF;
    --accent-hover: #7BA1FF;
    --accent-soft: rgba(91,140,255,0.16);
    --accent-2: #FBBF24;
    --success: #33D690;
    --success-soft: rgba(51,214,144,0.16);
    --warn-soft: rgba(251,191,36,0.16);
    --danger: #FF6B5E;
    --danger-soft: rgba(255,107,94,0.16);
    --shadow: 0 1px 2px rgba(0,0,0,0.35), 0 4px 18px rgba(0,0,0,0.3);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] { background: var(--bg-panel); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 13px !important; font-weight: 700 !important; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--text-soft) !important; margin-top: 20px !important;
    padding-bottom: 6px; border-bottom: 1px solid var(--border);
}
[data-testid="stSidebar"] hr { border-color: var(--border); }

.app-header { margin-bottom: 22px; }
.app-title { font-size: 30px; font-weight: 800; color: var(--text); display:flex; align-items:center; gap:10px; letter-spacing:-0.01em; }
.app-tagline { font-size: 14.5px; color: var(--text-soft); margin-top: 4px; }
.app-byline { font-size: 12.5px; color: var(--accent); font-weight: 600; margin-top: 8px; display:inline-block; background: var(--accent-soft); padding: 3px 10px; border-radius: 20px; }

.stButton > button { border-radius: 8px; font-weight: 600; font-size: 13.5px; border: 1px solid var(--border); transition: all 0.15s ease; }
.stButton > button[kind="primary"] { background: var(--accent); color: #FFFFFF; border: 1px solid var(--accent); }
.stButton > button[kind="primary"]:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
@media (prefers-reduced-motion: reduce) { .stButton > button { transition: none; } }

.info-card { background: var(--bg-panel); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 10px; padding: 14px 18px; box-shadow: var(--shadow); margin-bottom: 10px; }
.info-card .card-body { color: var(--text-soft); font-size: 13.5px; line-height: 1.7; margin-top: 4px; }

.vitals-row { display:flex; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
.vital-tile { flex:1; min-width:190px; background: var(--bg-panel); border:1px solid var(--border); border-radius: 12px; padding: 16px 18px; box-shadow: var(--shadow); }
.vital-label { font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-soft); display:flex; align-items:center; gap:7px; }
.vital-dot { width:8px; height:8px; border-radius:50%; display:inline-block; box-shadow: 0 0 0 3px var(--accent-soft); }
.vital-value { font-size: 21px; font-weight: 800; color: var(--text); margin-top: 6px; }
.vital-sub { font-size: 11.5px; color: var(--text-soft); margin-top: 2px; font-family:'JetBrains Mono', monospace; }

.feature-card { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 14px; padding: 20px 20px 18px 20px; height: 100%; box-shadow: var(--shadow); }
.feature-card .icon { width: 38px; height: 38px; border-radius: 10px; background: var(--accent-soft); color: var(--accent); display:flex; align-items:center; justify-content:center; font-size: 18px; margin-bottom: 12px; }
.feature-card h4 { font-size: 16px; margin: 0 0 6px 0; font-weight: 700; }
.feature-card p { font-size: 13.5px; color: var(--text-soft); margin:0; line-height:1.6; }

.hero-panel { background: linear-gradient(135deg, var(--accent-soft), var(--bg-panel) 60%); border: 1px solid var(--border); border-radius: 16px; padding: 30px 32px; margin-bottom: 24px; }
.hero-eyebrow { font-size: 12px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--accent); }
.hero-title { font-size: 25px; font-weight: 800; margin: 8px 0 8px 0; max-width: 640px; color: var(--text); }
.hero-desc { font-size: 14px; color: var(--text-soft); max-width: 580px; line-height: 1.65; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 13.5px; color: var(--text-soft); padding: 10px 18px; background: transparent; border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; background: var(--accent-soft) !important; border-bottom: 2px solid var(--accent) !important; }

[data-testid="stMetricValue"] { color: var(--text); font-weight: 800; }
[data-testid="stMetricLabel"] { color: var(--text-soft); font-weight: 600; font-size: 12px !important; }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }

.section-title { font-size: 15px; font-weight: 700; color: var(--text); margin-bottom: 2px; }
.app-footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border); text-align: center; color: var(--text-soft); font-size: 12.5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# Chart palette — works on both light and dark backgrounds: saturated
# hues + transparent chart backgrounds so the page's own surface shows
# through for line/scatter charts. Heatmaps use an OPAQUE panel color
# instead (see chart_layout(opaque=True)) since a transparent midpoint
# makes near-zero cells vanish into the page background.
# ==========================================================
CHART_PALETTE = ["#2F5FE0", "#F97066", "#14B8A6", "#F5B942", "#8B7CF6", "#37B6E0"]
GRID_COLOR = "rgba(120,128,140,0.20)"
AXIS_TEXT_COLOR = "rgba(120,128,140,0.95)"
HEATMAP_PANEL_BG = "#F4F6F9"  # fixed, readable regardless of page theme

STATUS_COLOR = {"poor": "#E4483A", "fair": "#F5B942", "good": "#0FA968", "excellent": "#0FA968"}


def chart_layout(opaque=False, **overrides):
    layout = dict(
        font=dict(family="Inter, sans-serif", color=AXIS_TEXT_COLOR, size=12),
        paper_bgcolor=HEATMAP_PANEL_BG if opaque else "rgba(0,0,0,0)",
        plot_bgcolor=HEATMAP_PANEL_BG if opaque else "rgba(0,0,0,0)",
        margin=dict(l=48, r=24, t=40, b=44),
        colorway=CHART_PALETTE,
        legend=dict(font=dict(size=11)),
    )
    layout.update(overrides)
    return layout


def style_axes(fig):
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color=AXIS_TEXT_COLOR))
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color=AXIS_TEXT_COLOR))
    return fig


def format_loso_accuracy(metrics: dict) -> str:
    """metrics['loso_accuracy_mean'] can be None (no groups given, or every
    LOSO fold failed) — never do metrics.get('loso_accuracy_mean', 0) * 100
    directly, since .get()'s default only applies when the KEY is missing,
    not when its value is None, and None * 100 raises TypeError."""
    loso = metrics.get('loso_accuracy_mean')
    if loso is None:
        n_failed = metrics.get('loso_n_folds_failed', 0)
        return "N/A (all LOSO folds failed)" if n_failed else "N/A (no subject/group labels)"
    return f"{loso * 100:.1f}%"


def vital_tile_html(label, value, sub, status="neutral"):
    dot = STATUS_COLOR.get(status, "#2F5FE0")
    # NOTE: kept as ONE physical line — Streamlit's markdown renderer can
    # mis-render multi-line unsafe_allow_html blocks as literal text
    # (see github.com/streamlit/streamlit/issues/859), so every injected
    # HTML snippet in this file is built without embedded newlines.
    return (f'<div class="vital-tile"><div class="vital-label">'
            f'<span class="vital-dot" style="background:{dot}"></span>{label}</div>'
            f'<div class="vital-value">{value}</div>'
            f'<div class="vital-sub">{sub}</div></div>')


# ==========================================================
# Header — product name + author only, no version numbers
# ==========================================================
st.markdown(
    f'<div class="app-header">'
    f'<div class="app-title">🧬 MyoControl Suite</div>'
    f'<div class="app-tagline">EMG Signal Analysis · Gesture Classification · Clinical Reporting</div>'
    f'<div class="app-byline">Built by {AUTHOR}</div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ==========================================================
# Session state
# ==========================================================
for key, default in [
    ('engine_result', None), ('raw_signal', None), ('filtered_signal', None),
    ('classifier', None), ('acquisition_info', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ==========================================================
# Sidebar
# ==========================================================
with st.sidebar:
    st.header("Configuration")

    mode = st.radio("UI Mode", ["Advanced (Researcher)", "Simplified (Clinician)"], index=0)
    simplified = mode.startswith("Simplified")

    st.divider()
    st.subheader("Signal Source")
    signal_source = st.radio("Source", ["Simulation", "Upload File"], index=0, horizontal=True)

    raw = None
    if signal_source == "Simulation":
        sim_duration = st.slider("Duration (s)", 1.0, 30.0, 5.0, 0.5)
        sim_gesture = st.selectbox("Gesture", EMGSimulator().list_gestures(), index=1)
        sim_channels = st.slider("Channels", 1, 8, 4)
        sim_intensity = st.slider("Intensity", 0.1, 2.0, 1.0, 0.1)
    else:
        uploaded = st.file_uploader("EMG file (CSV, TXT, NPY, MAT)", type=['csv', 'txt', 'npy', 'mat'], accept_multiple_files=False)
        upload_labels = None
        if uploaded:
            try:
                if uploaded.name.endswith('.npy'):
                    raw = np.load(uploaded)
                    if raw.ndim == 1:
                        raw = raw.reshape(-1, 1)
                    st.success(f"Loaded: {raw.shape[0]} samples × {raw.shape[1]} channels")
                elif uploaded.name.endswith('.mat'):
                    fields_info = inspect_mat_fields(uploaded)
                    numeric_2d_fields = [k for k, v in fields_info.items() if v['is_numeric'] and v['ndim'] == 2]
                    other_fields = [k for k in fields_info if k not in numeric_2d_fields]
                    if other_fields:
                        st.caption(f"Non-numeric or non-2D fields skipped: {', '.join(other_fields)}")
                    if not numeric_2d_fields:
                        st.error("No plain 2D numeric fields found in this .mat file.")
                    else:
                        default_emg_field = 'emg' if 'emg' in numeric_2d_fields else numeric_2d_fields[0]
                        emg_field_selected = st.selectbox(
                            "EMG field", numeric_2d_fields, index=numeric_2d_fields.index(default_emg_field))
                        raw = load_mat_numeric_field(uploaded, emg_field_selected)
                        # NinaPro/Myo-style exports are usually (samples, channels); if the
                        # picked field looks transposed (far more columns than rows), flip it.
                        if raw.shape[0] < raw.shape[1] and raw.shape[1] > 64:
                            raw = raw.T
                        st.success(f"Loaded: {raw.shape[0]} samples × {raw.shape[1]} channels from '{emg_field_selected}'")

                        label_field_options = ["(none)"] + [k for k in numeric_2d_fields if k != emg_field_selected]
                        default_label_field = 'restimulus' if 'restimulus' in label_field_options else (
                            'stimulus' if 'stimulus' in label_field_options else '(none)')
                        label_field_selected = st.selectbox(
                            "Label field (optional, reference only here)", label_field_options,
                            index=label_field_options.index(default_label_field))
                        if label_field_selected != "(none)":
                            upload_labels = load_mat_numeric_field(uploaded, label_field_selected).flatten()
                            st.caption(f"Classes in '{label_field_selected}': {sorted(np.unique(upload_labels).tolist())} "
                                       "— to actually train on these labels, use this file from the Classification "
                                       "tab's 'Real: NinaPro .mat' option.")

                        st.info("Sampling rate isn't stored in .mat files reliably — set it manually below "
                                "(NinaPro: DB1=100Hz, DB2/DB3=2000Hz, DB4=2000Hz, DB5=200Hz; verify against "
                                "your device's documentation).")
                else:
                    df_upload = robust_read_table(uploaded)
                    st.caption(f"Detected {df_upload.shape[0]} rows × {df_upload.shape[1]} columns: {', '.join(df_upload.columns)}")

                    non_channel_guess = {c for c in df_upload.columns if c.strip().lower() in
                                          ('time', 'timestamp', 't', 'class', 'label', 'gesture', 'id', 'subject')}
                    default_channels = [c for c in df_upload.columns if c not in non_channel_guess]

                    channel_cols_selected = st.multiselect(
                        "Channel columns", df_upload.columns.tolist(), default=default_channels)
                    remaining_cols = ["(none)"] + [c for c in df_upload.columns if c not in channel_cols_selected]
                    default_time_idx = remaining_cols.index('time') if 'time' in remaining_cols else 0
                    time_col_selected = st.selectbox("Time column (optional, for sampling-rate estimate)",
                                                      remaining_cols, index=default_time_idx)
                    default_label_idx = remaining_cols.index('class') if 'class' in remaining_cols else 0
                    label_col_selected = st.selectbox("Label/class column (optional, reference only here)",
                                                       remaining_cols, index=default_label_idx)
                    time_unit_selected = "ms"
                    if time_col_selected != "(none)":
                        time_unit_selected = st.radio("Time column unit", ["ms", "s"], horizontal=True)

                    if not channel_cols_selected:
                        st.warning("Select at least one channel column.")
                    else:
                        try:
                            raw = df_upload[channel_cols_selected].to_numpy(dtype=float)
                            st.success(f"Loaded: {raw.shape[0]} samples × {raw.shape[1]} channels")

                            if label_col_selected != "(none)":
                                upload_labels = df_upload[label_col_selected].to_numpy()
                                st.caption(f"Classes in '{label_col_selected}': {sorted(np.unique(upload_labels).tolist())} "
                                           "— to actually train on these labels, use this file from the Classification "
                                           "tab's 'Real CSV/TXT dataset (raw + label column)' option.")

                            if time_col_selected != "(none)":
                                t_numeric = df_upload[time_col_selected].to_numpy(dtype=float)
                                dt_raw = np.median(np.diff(t_numeric))
                                dup_frac = float((raw[:-1] == raw[1:]).all(axis=1).mean()) if len(raw) > 1 else 0.0
                                if dt_raw > 0:
                                    dt_seconds = dt_raw / 1000.0 if time_unit_selected == "ms" else dt_raw
                                    suggested_fs = int(round(1.0 / dt_seconds))
                                    if 20 <= suggested_fs <= 10000:
                                        st.session_state['suggested_fs'] = suggested_fs
                                        msg = f"Estimated sampling rate from '{time_col_selected}': **{suggested_fs} Hz** (pre-filled below)."
                                        if dup_frac > 0.5:
                                            msg += (f" ⚠️ {dup_frac*100:.0f}% of consecutive rows have identical channel "
                                                    f"values — this timestamp likely reflects a logging clock, not the "
                                                    f"device's true sample rate (e.g. the Myo armband logs on a 1ms clock "
                                                    f"but its documented native EMG rate is ~200 Hz, with readings held/"
                                                    f"repeated in between). Verify against your device's actual spec "
                                                    f"before trusting this number.")
                                        st.info(msg)
                                    else:
                                        st.warning(f"Estimated sampling rate ({suggested_fs} Hz) is outside a plausible "
                                                   "EMG range — check the time unit, or set Sampling rate manually below.")
                        except ValueError as e:
                            st.error(f"Selected channel columns aren't all numeric: {e}")
                            raw = None
            except Exception as e:
                st.error(f"Load failed: {e}")

    if not simplified:
        st.divider()
        st.subheader("Performance")
        downsample = st.checkbox("Downsample (large files)", value=False)
        max_samples = st.number_input("Max samples/channel", 1000, 1_000_000, 30000, step=1000)
        benchmark_mode = st.checkbox("Benchmark mode", value=True)
        compute_freq = st.checkbox("Compute frequency features (MDF, MNF)", value=False)

        st.subheader("Filters")
        filter_type = st.selectbox("Filter type", ['butterworth', 'chebyshev', 'bessel', 'elliptic'], index=0)
        noise_method = st.selectbox("Noise estimation", ['percentile', 'median', 'manual'], index=0)
        psd_method = st.selectbox("PSD method", ['welch', 'fft'], index=0)
        chunk_duration = st.number_input("Chunk duration (s, 0=off)", 0.0, 60.0, 0.0, step=0.5)

        st.subheader("Signal Parameters")
        _suggested_fs = st.session_state.get('suggested_fs', 2000)
        sampling_rate = st.slider("Sampling rate (Hz)", 100, 4000, min(max(_suggested_fs, 100), 4000), 50)
        cutoff_low = st.slider("High-pass (Hz)", 5.0, 50.0, 20.0, 1.0)
        cutoff_high = st.slider("Low-pass (Hz)", 200.0, 500.0, 450.0, 10.0)
        filter_order = st.slider("Filter order", 2, 8, 4)
        notch_freq = st.slider("Notch (Hz)", 50.0, 60.0, 50.0, 1.0)

        st.subheader("Windowing")
        window_ms = st.slider("Window (ms)", 50, 300, 100, 10)
        overlap = st.slider("Overlap", 0.0, 0.9, 0.5, 0.05)
    else:
        sampling_rate = 2000
        cutoff_low, cutoff_high = 20.0, 450.0
        filter_order, notch_freq = 4, 50.0
        window_ms, overlap = 100, 0.5
        filter_type = 'butterworth'
        noise_method = 'percentile'
        psd_method = 'welch'
        chunk_duration = 0
        compute_freq = False
        benchmark_mode = False

    st.divider()

    if st.button("▶ Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                config = EMGConfig(
                    sampling_rate=sampling_rate, cutoff_low=cutoff_low, cutoff_high=cutoff_high,
                    filter_order=filter_order, notch_freq=notch_freq,
                    window_size=int(window_ms * sampling_rate / 1000), overlap=overlap,
                    filter_type=filter_type, noise_estimation_method=noise_method,
                    psd_method=psd_method, chunk_duration=chunk_duration if chunk_duration > 0 else None,
                    compute_frequency_features=compute_freq,
                )
                engine = EMGEngine(config)

                if signal_source == "Simulation":
                    sim = EMGSimulator(fs=sampling_rate, n_channels=sim_channels)
                    raw_signal = sim.generate_contraction(sim_duration, sim_gesture, intensity_scale=sim_intensity)
                    acquisition_info = {'source': 'Simulation', 'gesture': sim_gesture, 'intensity': sim_intensity}
                else:
                    if raw is None:
                        st.error("Please upload a file first.")
                        st.stop()
                    raw_signal = raw
                    acquisition_info = {'source': f"Upload ({uploaded.name})", 'gesture': None, 'intensity': None}

                if not simplified and downsample and len(raw_signal) > max_samples:
                    step = len(raw_signal) // max_samples
                    raw_signal = raw_signal[::step]

                result = engine.process(
                    raw_signal, selected_channel=0, measure_time=benchmark_mode,
                    compute_freq_features=compute_freq,
                )

                st.session_state.raw_signal = raw_signal
                st.session_state.engine_result = result
                st.session_state.config = config
                st.session_state.filtered_signal = engine.preprocess(raw_signal)
                st.session_state.acquisition_info = acquisition_info

                try:
                    st.session_state.last_session_id = db.save_session(
                        result, config, acquisition_info=acquisition_info)
                except Exception as db_err:
                    st.session_state.last_session_id = None
                    st.warning(f"Session not saved to history: {db_err}")

                st.success("Analysis complete.")
            except Exception as e:
                st.error(f"Processing failed: {e}")

    if st.session_state.engine_result:
        st.divider()
        st.subheader("Export")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export JSON", use_container_width=True):
                st.download_button(
                    "Download JSON",
                    data=json.dumps(st.session_state.engine_result, indent=2, default=str),
                    file_name=f"emg_analysis_{int(time.time())}.json", mime="application/json",
                )
        with col2:
            if st.button("Export CSV", use_container_width=True):
                feat_rows = []
                ts = st.session_state.engine_result['time_series']
                for ch_idx, ch_feats in enumerate(ts['features']):
                    for i, f in enumerate(ch_feats):
                        feat_rows.append({'channel': ch_idx, 'window': i, **f})
                df = pd.DataFrame(feat_rows)
                st.download_button(
                    "Download CSV", data=df.to_csv(index=False),
                    file_name=f"emg_features_{int(time.time())}.csv", mime="text/csv",
                )
        if st.button("📄 Export PDF Report", use_container_width=True):
            with st.spinner("Building PDF..."):
                try:
                    pdf_bytes = generate_pdf_report(
                        st.session_state.engine_result,
                        raw_signal=st.session_state.raw_signal,
                        filtered_signal=st.session_state.filtered_signal,
                        acquisition_info=st.session_state.acquisition_info,
                        classifier_metrics=st.session_state.get('classifier_metrics'),
                        author=AUTHOR,
                    )
                    st.download_button(
                        "Download PDF", data=pdf_bytes,
                        file_name=f"emg_report_{int(time.time())}.pdf", mime="application/pdf",
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")


# ==========================================================
# Main panel — landing / empty state
# ==========================================================
if st.session_state.engine_result is None:
    st.markdown(
        '<div class="hero-panel">'
        '<div class="hero-eyebrow">Zero-calibration · CPU-only · Cross-subject</div>'
        '<div class="hero-title">A research-grade EMG console, without the GPU rack or the price tag.</div>'
        '<div class="hero-desc">Configure acquisition on the left, then run analysis on a simulated '
        'contraction or an uploaded recording. The dashboard reports signal quality, decoded gestures, '
        'and per-window feature evidence — with full transparency into every number it shows you.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="feature-card"><div class="icon">🔬</div>'
            '<h4>Research-grade filtering</h4>'
            '<p>IEEE / ISEK-compliant bandpass and notch filters, 7 time-domain '
            'and 3 frequency-domain features per window.</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="feature-card"><div class="icon">🤖</div>'
            '<h4>Gesture classification</h4>'
            '<p>Train XGBoost, LDA, or Random Forest classifiers on your EMG '
            'data and inspect each decision with SHAP.</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="feature-card"><div class="icon">📋</div>'
            '<h4>Clinical reporting</h4>'
            '<p>Muscle activity, fatigue index, artifact detection, and full '
            'JSON / CSV export for downstream analysis.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="app-footer">MyoControl Suite &nbsp;·&nbsp; Built by {AUTHOR}</div>', unsafe_allow_html=True)
    st.stop()

result = st.session_state.engine_result
acq = st.session_state.acquisition_info or {}

# ==========================================================
# Vitals bar — row 1: clinical signal readout
# ==========================================================
quality = result['signal_quality']['snr_quality']
activity = result['clinical_interpretation']['muscle_activity']
snr = result['signal_quality']['mean_snr_db']
n_ch = result['metadata']['n_channels']
n_w = result['summary_statistics'].get('channel_0', {}).get('n_windows', 0)

vitals_html = '<div class="vitals-row">'
vitals_html += vital_tile_html("Signal Quality", quality.title(), f"SNR {snr:.1f} dB", quality)
vitals_html += vital_tile_html("Muscle Activity", activity.split('_')[0].title(), "Clinical interpretation",
                                "good" if "active" in activity.lower() else "neutral")
vitals_html += vital_tile_html("SNR", f"{snr:.1f} dB", "Mean, channel 0", quality)
vitals_html += vital_tile_html("Channels / Windows", f"{n_ch} / {n_w}", "Acquisition geometry", "neutral")
vitals_html += '</div>'
st.markdown(vitals_html, unsafe_allow_html=True)

# Row 2: acquisition / contraction parameters (what was actually recorded)
gesture_label = acq.get('gesture') or "—"
intensity_val = acq.get('intensity')
intensity_label = f"{intensity_val:.1f}×" if intensity_val is not None else "—"
duration_s = result['metadata']['duration_seconds']
n_samples = result['metadata']['n_samples']
fs = result['metadata']['sampling_rate']

vitals_html_2 = '<div class="vitals-row">'
vitals_html_2 += vital_tile_html("Contraction Type", gesture_label.title() if gesture_label != "—" else gesture_label,
                                  acq.get('source', '—'), "neutral")
vitals_html_2 += vital_tile_html("Contraction Intensity", intensity_label, "Simulator scale factor", "neutral")
vitals_html_2 += vital_tile_html("Sampling Rate", f"{fs} Hz", f"{result['metadata']['filter_config'].get('cutoff_low', 20):.0f}–{result['metadata']['filter_config'].get('cutoff_high', 450):.0f} Hz band", "neutral")
vitals_html_2 += vital_tile_html("Duration / Samples", f"{duration_s:.2f}s / {n_samples}", "Acquired signal length", "neutral")
vitals_html_2 += '</div>'
st.markdown(vitals_html_2, unsafe_allow_html=True)

# ==========================================================
# Tabs
# ==========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Signal Analysis", "📊 Feature Extraction", "📉 Spectral Analysis",
    "🤖 Classification", "📋 Technical Report", "📐 Statistics", "🗄️ History",
])

# --------------------------------------------------------
# Tab 1: Signal Analysis
# --------------------------------------------------------
with tab1:
    st.markdown('<div class="section-title">Raw vs Filtered vs RMS Envelope</div>', unsafe_allow_html=True)
    raw_sig = st.session_state.raw_signal
    filtered = st.session_state.filtered_signal
    if raw_sig is None or filtered is None:
        st.info("Raw/filtered waveforms aren't available for this session (reloaded from history, which "
                "stores only the extracted result). Run a new analysis to see this chart.")
    else:
        sr = st.session_state.config.sampling_rate
        t = np.arange(len(raw_sig)) / sr

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                             subplot_titles=("Raw EMG (Channel 0)", "Filtered EMG (Channel 0)", "RMS Activation Envelope"))
        fig.add_trace(go.Scatter(x=t, y=raw_sig[:, 0], name="Raw", line=dict(color="rgba(47,95,224,0.35)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=filtered[:, 0], name="Filtered", line=dict(color=CHART_PALETTE[0], width=1.3)), row=2, col=1)

        win = st.session_state.config.window_size
        step = max(1, int(win * (1 - st.session_state.config.overlap)))
        rms_env, rms_t = [], []
        for i in range(0, len(filtered) - win, step):
            rms_env.append(np.sqrt(np.mean(filtered[i:i + win, 0] ** 2)))
            rms_t.append(i / sr)
        fig.add_trace(go.Scatter(x=rms_t, y=rms_env, name="RMS Envelope", line=dict(color=CHART_PALETTE[2], width=2.4),
                                  fill='tozeroy', fillcolor="rgba(20,184,166,0.12)"), row=3, col=1)

        fig.update_layout(**chart_layout(height=680, showlegend=False))
        fig = style_axes(fig)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------
# Tab 2: Feature Extraction
# --------------------------------------------------------
with tab2:
    st.markdown('<div class="section-title">Time-Domain Features (per window)</div>', unsafe_allow_html=True)
    ts = result['time_series']
    feats = ts['features'][0]
    if feats:
        df = pd.DataFrame(feats)
        df.insert(0, 'window', range(len(df)))
        df.insert(0, 'time_s', [round(tt, 3) for tt in ts['timestamps']])
        st.dataframe(df, use_container_width=True)

        feature_cols = [c for c in df.columns if c in TIME_FEATURES]
        selected = st.multiselect("Features to plot", feature_cols,
                                   default=['MAV', 'RMS'] if 'MAV' in feature_cols else feature_cols[:2])
        if selected:
            fig = make_subplots(rows=len(selected), cols=1, shared_xaxes=True, subplot_titles=selected)
            for i, feat in enumerate(selected, 1):
                fig.add_trace(go.Scatter(x=df['time_s'], y=df[feat], name=feat,
                                          line=dict(width=2, color=CHART_PALETTE[(i - 1) % len(CHART_PALETTE)])), row=i, col=1)
            fig.update_layout(**chart_layout(height=200 * len(selected), showlegend=False))
            fig = style_axes(fig)
            fig.update_xaxes(title_text="Time (s)", row=len(selected), col=1)
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------
# Tab 3: Spectral Analysis
# --------------------------------------------------------
with tab3:
    st.markdown('<div class="section-title">Frequency Spectrum</div>', unsafe_allow_html=True)
    filtered = st.session_state.filtered_signal
    if filtered is None:
        st.info("Filtered waveform isn't available for this session (reloaded from history, which "
                "stores only the extracted result). Run a new analysis to see the spectrum.")
    else:
        sr = st.session_state.config.sampling_rate

        ch0 = filtered[:, 0]
        n = len(ch0)
        fft_vals = np.fft.rfft(ch0)
        fft_mag = np.abs(fft_vals) / n
        freqs = np.fft.rfftfreq(n, 1 / sr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs, y=fft_mag, name='FFT Magnitude', line=dict(color=CHART_PALETTE[0], width=1.6),
                                  fill='tozeroy', fillcolor="rgba(47,95,224,0.10)"))
        fig.add_vrect(x0=20, x1=450, fillcolor=CHART_PALETTE[2], opacity=0.08, annotation_text="EMG band (20-450 Hz)",
                      annotation_font=dict(color=AXIS_TEXT_COLOR, size=10))
        fig.add_vrect(x0=0, x1=20, fillcolor=CHART_PALETTE[1], opacity=0.10, annotation_text="Motion artifact",
                      annotation_font=dict(color=AXIS_TEXT_COLOR, size=10))
        fig.add_vline(x=50, line_dash="dash", line_color=CHART_PALETTE[1], annotation_text="50 Hz powerline",
                      annotation_font=dict(color=CHART_PALETTE[1], size=10))
        fig.update_layout(**chart_layout(xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
                                          xaxis_range=[0, min(sr / 2, 500)], height=450))
        fig = style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

    if result['metadata']['filter_config'].get('compute_frequency_features'):
        st.markdown('<div class="section-title">Frequency-Domain Features Over Time</div>', unsafe_allow_html=True)
        if 'freq_features' in ts and ts['freq_features']:
            ff = ts['freq_features'][0]
            if ff:
                df_f = pd.DataFrame(ff)
                df_f.insert(0, 'time_s', [round(tt, 3) for tt in ts['timestamps']])
                st.dataframe(df_f, use_container_width=True, hide_index=True)

# --------------------------------------------------------
# Tab 4: Classification
# --------------------------------------------------------
with tab4:
    st.markdown('<div class="section-title">Gesture Classification</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown("**Model**")
        model_type = st.selectbox("Algorithm", ['xgboost', 'random_forest', 'lda', 'linear_svc'], index=0)

        data_source = st.radio("Training data", ["Synthetic (demo)", "Real: pre-extracted features CSV",
                                                   "Real: raw signal + label column", "Real: NinaPro .mat"], index=0)

        if data_source == "Synthetic (demo)":
            n_classes = st.slider("Classes (synthetic)", 2, 5, 5)
            n_samples_cls = st.slider("Samples per class", 20, 200, 50, step=10)
            st.caption("⚠️ Synthetic data only demonstrates the pipeline — it is not evidence the "
                       "classifier works on real muscle signals.")

            if st.button("🎯 Train Demo Classifier", type="primary"):
                with st.spinner(f"Training {model_type} on synthetic data..."):
                    try:
                        dataset = generate_synthetic_emg_dataset(n_classes=n_classes, n_samples_per_class=n_samples_cls)
                        clf = EMGClassifier(model_type=model_type)
                        X = clf.prepare_features(dataset['features'])
                        metrics = clf.fit(X, dataset['labels'], groups=dataset['groups'])
                        st.session_state.classifier = clf
                        st.session_state.classifier_metrics = metrics
                        st.success(f"Trained! LOSO accuracy: {format_loso_accuracy(metrics)}")
                    except Exception as e:
                        st.error(f"Training failed: {e}")

        elif data_source == "Real: pre-extracted features CSV":
            st.caption("Columns required: MAV, RMS, ZCR, WL, SSC, WAMP, MYOP, label, and (for LOSO) subject.")
            csv_file = st.file_uploader("Feature CSV/TXT", type=['csv', 'txt'], key="real_csv_uploader")
            label_col = st.text_input("Label column", value="label")
            group_col = st.text_input("Subject/group column (optional, enables LOSO)", value="subject")

            if st.button("🎯 Train on Feature CSV", type="primary"):
                if csv_file is None:
                    st.error("Upload a file first.")
                else:
                    with st.spinner(f"Training {model_type} on uploaded data..."):
                        try:
                            ds = load_features_csv(csv_file, label_col=label_col, group_col=group_col or None)
                            clf = EMGClassifier(model_type=model_type)
                            X = clf.prepare_features(ds['features'])
                            metrics = clf.fit(X, ds['labels'], groups=ds['groups'])
                            st.session_state.classifier = clf
                            st.session_state.classifier_metrics = metrics
                            if ds['groups'] is None:
                                st.success(f"Trained on {len(ds['labels'])} real windows. Train accuracy: "
                                           f"{metrics['train_accuracy'] * 100:.1f}%. No subject/group column, "
                                           "so LOSO cross-subject accuracy wasn't computed — train accuracy "
                                           "alone overstates real-world performance.")
                            else:
                                st.success(f"Trained on {len(ds['labels'])} real windows across "
                                           f"{metrics.get('loso_n_folds', '?')} subjects. LOSO accuracy: "
                                           f"{format_loso_accuracy(metrics)}")
                        except Exception as e:
                            st.error(f"Training failed: {e}")

        elif data_source == "Real: raw signal + label column":
            st.caption("For files with ONE ROW PER SAMPLE: channel columns + a label column repeated on every "
                       "row (e.g. the UCI 'EMG data for gestures' Myo-armband format: time, channel1-8, class). "
                       "Auto-detects delimiter and header. Upload several files at once (one per subject) to "
                       "get a real cross-subject LOSO estimate instead of train-accuracy-only.")
            raw_files = st.file_uploader("Raw signal + label file(s) (CSV/TXT)", type=['csv', 'txt'],
                                          accept_multiple_files=True, key="real_raw_uploader")

            if raw_files:
                st.caption(f"{len(raw_files)} file(s) selected: {', '.join(f.name for f in raw_files)}")
                try:
                    df_preview = robust_read_table(raw_files[0])
                    st.caption(f"Columns detected from '{raw_files[0].name}' (applied to all selected files): "
                               f"{df_preview.shape[0]} rows × {df_preview.shape[1]} columns: {', '.join(df_preview.columns)}")

                    default_label = 'class' if 'class' in df_preview.columns else ('label' if 'label' in df_preview.columns else df_preview.columns[-1])
                    label_col_r = st.selectbox("Label column", df_preview.columns.tolist(),
                                                index=df_preview.columns.tolist().index(default_label))
                    time_options = ["(none — specify fs manually)"] + [c for c in df_preview.columns if c != label_col_r]
                    default_time = 'time' if 'time' in time_options else time_options[0]
                    time_col_r = st.selectbox("Time column", time_options, index=time_options.index(default_time))

                    fs_r = None
                    if time_col_r != "(none — specify fs manually)":
                        time_unit_r = st.radio("Time unit", ["ms", "s"], horizontal=True, key="raw_time_unit")
                    else:
                        fs_r = st.number_input("Sampling rate (Hz)", 20, 10000, 200, step=10)
                        time_unit_r = "ms"

                    channel_options = [c for c in df_preview.columns if c not in (label_col_r, time_col_r)]
                    channel_cols_r = st.multiselect("Channel columns", channel_options, default=channel_options)
                    window_ms_r = st.slider("Window (ms)", 20, 500, 100, 10, key="raw_window_ms")
                    if len(raw_files) == 1:
                        subject_id_r = st.text_input("Subject/group ID for this file", value=Path(raw_files[0].name).stem)
                    else:
                        st.caption("Subject ID for each file is auto-assigned from its filename.")
                except Exception as e:
                    st.error(f"Couldn't read file: {e}")
                    df_preview = None

                if st.button("🎯 Train on Raw + Label File(s)", type="primary"):
                    with st.spinner(f"Extracting features from {len(raw_files)} file(s) and training {model_type}..."):
                        datasets = []
                        failed = []
                        for f in raw_files:
                            try:
                                f.seek(0)
                                sid = subject_id_r if len(raw_files) == 1 else Path(f.name).stem
                                ds_i = load_raw_labeled_table(
                                    f, label_col=label_col_r,
                                    time_col=(time_col_r if time_col_r != "(none — specify fs manually)" else None),
                                    time_unit=time_unit_r, channel_cols=channel_cols_r or None,
                                    fs=fs_r, window_ms=window_ms_r, subject_id=sid,
                                )
                                datasets.append(ds_i)
                            except Exception as e:
                                failed.append((f.name, str(e)))

                        if failed:
                            for fname, err in failed:
                                st.warning(f"Skipped '{fname}': {err}")

                        if not datasets:
                            st.error("No files loaded successfully — nothing to train on.")
                        else:
                            try:
                                merged = concatenate_subjects(datasets)
                                fs_used = {d['fs_estimated'] for d in datasets}
                                st.caption(f"Used sampling rate: {sorted(fs_used)} Hz — verify this matches your "
                                           "device's documented rate before trusting frequency-related results.")
                                clf = EMGClassifier(model_type=model_type)
                                X = clf.prepare_features(merged['features'])
                                metrics = clf.fit(X, merged['labels'], groups=merged['groups'])
                                st.session_state.classifier = clf
                                st.session_state.classifier_metrics = metrics
                                n_subjects = len(np.unique(merged['groups']))
                                if n_subjects > 1:
                                    st.success(f"Trained on {len(merged['labels'])} windows across {n_subjects} "
                                               f"subjects ({len(datasets)} file(s) loaded). LOSO accuracy: "
                                               f"{format_loso_accuracy(metrics)}")
                                else:
                                    st.success(f"Trained on {len(merged['labels'])} windows. Train accuracy: "
                                               f"{metrics['train_accuracy'] * 100:.1f}%. Single subject, so LOSO "
                                               "wasn't computed — upload files from more subjects for a real "
                                               "cross-subject estimate (train accuracy alone overstates performance).")
                            except Exception as e:
                                st.error(f"Training failed: {e}")

        else:  # Real: NinaPro .mat
            st.caption("For NinaPro .mat files (ninapro.hevs.ch): expects 'emg' and 'restimulus' "
                       "(falls back to 'stimulus') fields. fs is NOT auto-detected — NinaPro databases use "
                       "different hardware at different rates (DB1=100Hz, DB2/DB3=2000Hz, DB4=2000Hz, "
                       "DB5=200Hz — verify yours against the official docs). Upload several files at once "
                       "(one per subject) for a real cross-subject LOSO estimate.")
            mat_files = st.file_uploader("NinaPro .mat file(s)", type=['mat'],
                                          accept_multiple_files=True, key="ninapro_uploader")

            if mat_files:
                st.caption(f"{len(mat_files)} file(s) selected: {', '.join(f.name for f in mat_files)}")
                fs_np = st.number_input("Sampling rate (Hz) — required, see note above", 20, 10000, 100, step=10)
                window_ms_np = st.slider("Window (ms)", 20, 500, 200, 10, key="ninapro_window_ms")
                label_field_np = st.selectbox("Label field", ["restimulus", "stimulus"], index=0)

                if st.button("🎯 Train on NinaPro .mat File(s)", type="primary"):
                    with st.spinner(f"Extracting features from {len(mat_files)} file(s) and training {model_type}..."):
                        datasets = []
                        failed = []
                        for f in mat_files:
                            try:
                                f.seek(0)
                                sid = Path(f.name).stem
                                ds_i = load_ninapro_mat(
                                    f, fs=fs_np, label_field=label_field_np,
                                    window_ms=window_ms_np, subject_id=sid,
                                )
                                datasets.append(ds_i)
                            except Exception as e:
                                failed.append((f.name, str(e)))

                        if failed:
                            for fname, err in failed:
                                st.warning(f"Skipped '{fname}': {err}")

                        if not datasets:
                            st.error("No files loaded successfully — nothing to train on.")
                        else:
                            try:
                                merged = concatenate_subjects(datasets)
                                clf = EMGClassifier(model_type=model_type)
                                X = clf.prepare_features(merged['features'])
                                metrics = clf.fit(X, merged['labels'], groups=merged['groups'])
                                st.session_state.classifier = clf
                                st.session_state.classifier_metrics = metrics
                                n_subjects = len(np.unique(merged['groups']))
                                if n_subjects > 1:
                                    st.success(f"Trained on {len(merged['labels'])} windows across {n_subjects} "
                                               f"subjects ({len(datasets)} file(s) loaded, label field "
                                               f"'{datasets[0]['label_field_used']}'). LOSO accuracy: "
                                               f"{format_loso_accuracy(metrics)}")
                                else:
                                    st.success(f"Trained on {len(merged['labels'])} windows. Train accuracy: "
                                               f"{metrics['train_accuracy'] * 100:.1f}%. Single subject, so LOSO "
                                               "wasn't computed — upload files from more subjects for a real "
                                               "cross-subject estimate (train accuracy alone overstates performance).")
                            except Exception as e:
                                st.error(f"Training failed: {e}")

    with col1:
        clf = st.session_state.classifier
        if clf is None:
            st.info("👈 Train a classifier first (or load default model).")
        else:
            metrics = st.session_state.classifier_metrics
            st.markdown(f"**Model:** `{clf.model_type}` | **Classes:** {len(clf.classes_)} | **Features:** {len(clf.feature_names)}")

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Train Accuracy", f"{metrics['train_accuracy'] * 100:.1f}%")
            with mc2:
                loso = metrics.get('loso_accuracy_mean')
                n_failed = metrics.get('loso_n_folds_failed', 0)
                if loso is not None:
                    ci95 = metrics.get('loso_accuracy_ci95')
                    n_folds = metrics.get('loso_n_folds')
                    delta = None
                    if ci95:
                        delta = f"95% CI [{ci95[0]*100:.1f}, {ci95[1]*100:.1f}]% · n={n_folds} subjects"
                    st.metric("LOSO Accuracy (cross-subject)", f"{loso * 100:.1f}%", delta=delta, delta_color="off")
                    if n_failed > 0:
                        st.caption(f"⚠️ {n_failed} of {n_folds + n_failed} LOSO fold(s) failed and were excluded "
                                   "from this average — likely a held-out subject had a class not represented "
                                   "in the other subjects' training data. Try a smaller window (more windows "
                                   "per class per subject), or check class coverage per subject.")
                elif n_failed > 0:
                    st.metric("LOSO Accuracy (cross-subject)", "N/A")
                    st.caption(f"⚠️ All {n_failed} LOSO fold(s) failed — every held-out subject had at least one "
                               "class not represented in the remaining training subjects. Try a smaller window "
                               "(more windows per class per subject), or check class coverage per subject.")
                else:
                    st.metric("LOSO Accuracy (cross-subject)", "N/A")
                    st.caption("No subject/group labels were provided, so cross-subject accuracy couldn't be computed.")
            with mc3:
                st.metric("Samples", metrics['n_samples'])

            st.markdown("#### Predict on Current Signal")
            if st.button("🔮 Classify Current EMG"):
                feats = result['time_series']['features'][0]
                if feats:
                    preds = clf.predict(feats)

                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        ts_pred = result['time_series']['timestamps']
                        fig = go.Figure()
                        for i, cls in enumerate(clf.classes_):
                            mask = np.array(preds) == cls
                            if mask.any():
                                fig.add_trace(go.Scatter(x=np.array(ts_pred)[mask], y=np.array(preds)[mask], mode='markers',
                                                          name=cls, marker=dict(size=8, color=CHART_PALETTE[i % len(CHART_PALETTE)])))
                        fig.update_layout(**chart_layout(xaxis_title="Time (s)", yaxis_title="Predicted gesture", height=350))
                        fig = style_axes(fig)
                        st.plotly_chart(fig, use_container_width=True)

                    with col_b:
                        unique, counts = np.unique(preds, return_counts=True)
                        dist_df = pd.DataFrame({'gesture': unique, 'count': counts, 'percent': (counts / len(preds) * 100).round(1)})
                        st.markdown("**Distribution:**")
                        st.dataframe(dist_df, use_container_width=True, hide_index=True)
                        dominant = unique[np.argmax(counts)]
                        st.metric("Dominant gesture", dominant)

            st.markdown("#### Feature Importance (SHAP)")
            if st.button("💡 Explain Predictions"):
                if not SHAP_IMPORTABLE:
                    st.warning("SHAP is not installed in this Python environment. Install with: pip install shap")
                else:
                    with st.spinner("Computing SHAP values..."):
                        try:
                            explanation = clf.explain(max_samples=50)
                            imp = explanation['feature_importance']
                            df_imp = pd.DataFrame([{'feature': k, 'importance': v} for k, v in imp.items()])
                            fig = go.Figure(go.Bar(x=df_imp['importance'], y=df_imp['feature'], orientation='h',
                                                    marker_color=CHART_PALETTE[0]))
                            fig.update_layout(**chart_layout(xaxis_title="Mean |SHAP|", yaxis_title="Feature", height=350))
                            fig = style_axes(fig)
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"Explained {explanation['n_explained_samples']} of this model's own "
                                       f"training windows, using {explanation['explainer_type']}.")
                        except Exception as e:
                            st.error(f"SHAP explanation failed: {e}")

# --------------------------------------------------------
# Tab 5: Technical Report
# --------------------------------------------------------
with tab5:
    st.markdown('<div class="section-title">Clinical Interpretation</div>', unsafe_allow_html=True)

    ci = result['clinical_interpretation']
    sq = result['signal_quality']

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="info-card"><b>Muscle Activity:</b> <code>{ci["muscle_activity"]}</code>'
            f'<div class="card-body"><b>Activation Pattern:</b> <code>{ci["activation_pattern"]}</code><br>'
            f'<b>Signal Quality:</b> <code>{sq["snr_quality"]}</code> (SNR: {sq["mean_snr_db"]:.1f} dB)<br>'
            f'<b>Artifact Detected:</b> <code>{"Yes" if sq["artifact_detected"] else "No"}</code> '
            f'({sq["artifact_percentage"]:.2f}% of samples)</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        ch0_summary = result['summary_statistics'].get('channel_0', {})
        st.markdown(
            f'<div class="info-card"><b>Signal Characteristics</b>'
            f'<div class="card-body">Duration: {result["metadata"]["duration_seconds"]:.2f}s<br>'
            f'Samples: {result["metadata"]["n_samples"]}<br>'
            f'Channels: {result["metadata"]["n_channels"]}<br>'
            f'Windows: {ch0_summary.get("n_windows", 0)}<br>'
            f'Fatigue Index (Ch0): {ch0_summary.get("fatigue_index", 0):.4f}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<div class="section-title">Per-Channel Summary</div>', unsafe_allow_html=True)
    rows = [{'channel': ch_key, **ch_summary} for ch_key, ch_summary in result['summary_statistics'].items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<div class="section-title">Raw JSON Output</div>', unsafe_allow_html=True)
    with st.expander("View full JSON"):
        st.json(result)

# --------------------------------------------------------
# Tab 6: Statistics
# --------------------------------------------------------
with tab6:
    st.markdown('<div class="section-title">Statistical Analysis</div>', unsafe_allow_html=True)

    feats = result['time_series']['features'][0]
    if not feats:
        st.warning("No features available.")
    else:
        df = pd.DataFrame(feats)

        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        if len(df.columns) >= 2:
            st.markdown("#### Feature Correlation Matrix")
            zero_var_cols = df.columns[df.std() == 0].tolist()
            if zero_var_cols:
                st.caption(f"Excluded (zero variance in this window set): {', '.join(zero_var_cols)}")
            corr_cols = [c for c in df.columns if c not in zero_var_cols]
            corr = df[corr_cols].corr().fillna(0.0)
            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                text=np.round(corr.values, 2), texttemplate="%{text}",
                textfont=dict(size=11, color="#0F172A"),
                colorscale=[[0, "#F97066"], [0.5, "#F4F6F9"], [1, "#2F5FE0"]],
                zmid=0, zmin=-1, zmax=1,
            ))
            fig.update_layout(**chart_layout(opaque=True, height=500))
            st.plotly_chart(fig, use_container_width=True)

        if len(df.columns) >= 3:
            st.markdown("#### PCA (2 components)")
            from sklearn.decomposition import PCA
            pca_df = df.loc[:, df.std() > 0]
            if pca_df.shape[1] >= 2:
                pca = PCA(n_components=2)
                comps = pca.fit_transform(pca_df.values)
                fig = go.Figure(go.Scatter(
                    x=comps[:, 0], y=comps[:, 1], mode='markers',
                    marker=dict(size=8, color=np.arange(len(comps)), colorscale=[[0, "#8B7CF6"], [1, "#2F5FE0"]],
                                showscale=True, colorbar=dict(title="window")),
                ))
                fig.update_layout(**chart_layout(
                    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
                    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", height=450))
                fig = style_axes(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough non-constant features for PCA in this window set.")

# --------------------------------------------------------
# Tab 7: History — browse and reload past saved sessions
# --------------------------------------------------------
with tab7:
    st.markdown('<div class="section-title">Session History</div>', unsafe_allow_html=True)
    st.caption("Every completed analysis is saved locally (SQLite, no cloud) so you can revisit or compare past runs.")

    try:
        sessions = db.list_sessions(limit=100)
    except Exception as e:
        sessions = []
        st.error(f"Could not read session history: {e}")

    if not sessions:
        st.info("No saved sessions yet — run an analysis to start building history.")
    else:
        hist_df = pd.DataFrame(sessions)
        hist_df['created_at'] = pd.to_datetime(hist_df['created_at'], unit='s')
        display_cols = ['id', 'created_at', 'source', 'gesture', 'intensity',
                         'duration_seconds', 'snr_db', 'snr_quality', 'model_type', 'loso_accuracy_mean']
        display_cols = [c for c in display_cols if c in hist_df.columns]
        st.dataframe(hist_df[display_cols], use_container_width=True, hide_index=True)

        col_a, col_b = st.columns([1, 3])
        with col_a:
            selected_id = st.selectbox("Session ID to load", hist_df['id'].tolist())
            load_clicked = st.button("↻ Load this session")
            delete_clicked = st.button("🗑 Delete this session")

        if load_clicked:
            loaded = db.load_session(int(selected_id))
            if loaded:
                st.session_state.engine_result = loaded['result']
                st.session_state.acquisition_info = loaded['acquisition_info']
                try:
                    st.session_state.config = EMGConfig(**loaded['config'])
                except TypeError:
                    # stored config has fields EMGConfig no longer accepts
                    # (e.g. saved by an older app version) — fall back to
                    # defaults rather than crash the reload.
                    st.session_state.config = EMGConfig()
                # Raw/filtered waveforms aren't stored (only the processed
                # result is) — Signal Analysis / Spectral tabs show a note
                # instead of the waveform for reloaded sessions.
                st.session_state.raw_signal = None
                st.session_state.filtered_signal = None
                st.success(f"Session #{selected_id} loaded. Feature Extraction, Technical Report, and "
                           "Statistics use the saved result and work normally; Signal Analysis and Spectral "
                           "Analysis need the raw waveform, which isn't stored in history, so those two tabs "
                           "will show a note instead.")
                st.rerun()
            else:
                st.error("Session not found (it may have been deleted).")

        if delete_clicked:
            if db.delete_session(int(selected_id)):
                st.success(f"Session #{selected_id} deleted.")
                st.rerun()
            else:
                st.error("Delete failed — session not found.")

# ==========================================================
# Footer
# ==========================================================
st.markdown(f'<div class="app-footer">MyoControl Suite &nbsp;·&nbsp; Built by {AUTHOR}</div>', unsafe_allow_html=True)
