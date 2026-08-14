"""
dataset_loader.py - Adapters for training/evaluating EMGClassifier on REAL
labeled EMG data, instead of only generate_synthetic_emg_dataset().

This module is infrastructure, not validation: it cannot make the
classifier's reported accuracy scientifically meaningful by itself — that
still requires you to supply real, labeled, multi-subject EMG recordings
(e.g. your own pilot data, or a public corpus such as NinaPro
https://ninapro.hevs.ch/ or the Ottobock/BioPatRec datasets). What this
module does is get that raw data into the exact
{'features': ..., 'labels': ..., 'groups': ...} shape that
EMGClassifier.fit() / generate_synthetic_emg_dataset() already use, via
the SAME feature-extraction pipeline (extract_features_stream) the rest
of the app uses — so a real-data run and a synthetic-data run are
apples-to-apples comparable.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .features import extract_features_stream, estimate_noise_floor, TIME_FEATURES


def _window_and_extract_with_majority_label(
    raw_signal: np.ndarray,
    labels_per_sample: np.ndarray,
    fs: int,
    window_ms: int,
    overlap: float,
    threshold_multiplier: float,
    subject_id: Optional[str],
    channel: int = 0,
) -> Dict[str, np.ndarray]:
    """Shared core for load_raw_labeled_table() and load_ninapro_mat():
    window a raw (n_samples, n_channels) signal against a DENSE per-sample
    label array, extract features per window, and label each window by
    majority vote over the samples inside it. One implementation so the
    two loaders can't silently drift apart on this logic.
    """
    if raw_signal.ndim == 1:
        raw_signal = raw_signal.reshape(-1, 1)
    if len(labels_per_sample) != len(raw_signal):
        raise ValueError(
            f"Label array length ({len(labels_per_sample)}) doesn't match "
            f"signal length ({len(raw_signal)}).")

    window_size = max(2, int(window_ms * fs / 1000))
    step = max(1, int(window_size * (1 - overlap)))

    diff_noise_floor = estimate_noise_floor(
        np.abs(np.diff(raw_signal[:, channel])), sampling_rate=fs)
    threshold = threshold_multiplier * diff_noise_floor

    all_channel_features = extract_features_stream(
        raw_signal, fs, window_size=window_size, overlap=overlap,
        include_freq=False, threshold=threshold,
    )
    ch_features = all_channel_features[channel]

    window_labels = []
    for i in range(len(ch_features)):
        start = i * step
        end = start + window_size
        seg_labels = labels_per_sample[start:end]
        vals, counts = np.unique(seg_labels, return_counts=True)
        window_labels.append(vals[np.argmax(counts)])

    groups = np.array([subject_id] * len(ch_features)) if subject_id else None

    return {
        'features': ch_features,
        'labels': np.array(window_labels),
        'groups': groups,
        'fs_used': fs,
        'n_channels_used': raw_signal.shape[1],
    }


def robust_read_table(path_or_buffer) -> pd.DataFrame:
    """Read a CSV/TXT/TSV file whose delimiter and header aren't known in
    advance — the normal situation for EMG exports from different
    hardware/software (comma, tab, or whitespace separated; header
    present or not). Auto-detects the delimiter (engine='python',
    sep=None) and always treats the first row as a header — if your file
    genuinely has no header row, add one before loading; guessing at
    "is this a header or numeric data" silently is how you end up with a
    column literally named '1e-05'.
    """
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)
    df = pd.read_csv(path_or_buffer, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_raw_labeled_table(
    path_or_buffer,
    label_col: str = "class",
    time_col: Optional[str] = "time",
    time_unit: str = "ms",
    channel_cols: Optional[List[str]] = None,
    fs: Optional[int] = None,
    window_ms: int = 100,
    overlap: float = 0.5,
    threshold_multiplier: float = 1.0,
    subject_id: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load a RAW per-sample EMG table: one row per sample, one column per
    channel, plus a DENSE per-sample label column — the format most
    consumer EMG armbands/exports actually produce (e.g. an 8-channel
    export with a `class` value repeated on every row of a gesture — this
    is exactly the UCI "EMG data for gestures" Myo-bracelet format:
    Krilova et al. 2018, https://doi.org/10.24432/C5ZP5C). This is
    different from load_features_csv(), which expects one row per WINDOW
    with features already extracted.

    Each output window's label is the majority vote of the per-sample
    labels inside that window (a window straddling a label transition is
    labeled by whichever class covers more of it).

    fs auto-detection: if fs isn't given, it's estimated from the median
    step of `time_col`, interpreted according to time_unit ('ms' or 's').
    GET THIS WRONG AND EVERY DOWNSTREAM FEATURE IS GARBAGE: e.g. treating
    a millisecond column as seconds under-estimates fs by 1000x, which
    collapses the window size to almost nothing. If the resulting fs is
    outside a plausible surface-EMG range (20-10,000 Hz), this raises
    instead of silently continuing — pass fs explicitly if you hit that
    error and know the true rate.

    threshold_multiplier plays the same role as
    EMGConfig.feature_threshold_multiplier — the crossing threshold for
    WAMP/MYOP/ZCR/SSC is threshold_multiplier x the noise floor of the
    *differenced* channel-0 signal, estimated with the same method used
    everywhere else in this app (see EMGConfig docstring for why 0 is
    the wrong default).
    """
    df = robust_read_table(path_or_buffer)

    if label_col not in df.columns:
        raise ValueError(
            f"'{label_col}' column not found. Available columns: {list(df.columns)}")

    exclude = {label_col}
    if time_col and time_col in df.columns:
        exclude.add(time_col)
    if channel_cols is None:
        channel_cols = [c for c in df.columns if c not in exclude]
    missing = [c for c in channel_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Channel columns not found in file: {missing}")
    if not channel_cols:
        raise ValueError(
            "No channel columns left after excluding label/time columns — "
            "pass channel_cols explicitly.")

    try:
        raw_signal = df[channel_cols].to_numpy(dtype=float)
    except ValueError as e:
        raise ValueError(
            f"Could not convert channel columns {channel_cols} to numeric "
            f"— check you didn't include a non-numeric column (e.g. the "
            f"label or an id column) in channel_cols. Original error: {e}")
    labels_per_sample = df[label_col].to_numpy()

    if fs is None:
        if not (time_col and time_col in df.columns):
            raise ValueError(
                "fs not given and no time_col to estimate it from — pass fs explicitly.")
        dt_raw = np.median(np.diff(df[time_col].to_numpy(dtype=float)))
        if dt_raw <= 0:
            raise ValueError(
                f"Median step of '{time_col}' is {dt_raw} (non-positive) — "
                "can't estimate fs from it. Pass fs explicitly.")
        dt_seconds = dt_raw / 1000.0 if time_unit == "ms" else dt_raw
        fs = int(round(1.0 / dt_seconds))
        if not (20 <= fs <= 10000):
            raise ValueError(
                f"Auto-detected sampling rate ({fs} Hz, from median '{time_col}' "
                f"step of {dt_raw} {time_unit}) is outside a plausible surface-EMG "
                f"range (20-10,000 Hz) — time_unit is probably wrong (try the "
                f"other of 'ms'/'s'), or pass fs explicitly if you know the true rate.")

    window_result = _window_and_extract_with_majority_label(
        raw_signal, labels_per_sample, fs=fs, window_ms=window_ms,
        overlap=overlap, threshold_multiplier=threshold_multiplier,
        subject_id=subject_id, channel=0,
    )
    window_result['fs_estimated'] = window_result.pop('fs_used')
    window_result['channel_cols'] = channel_cols
    return window_result


def _load_mat(path_or_buffer):
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise ImportError(
            "scipy is required to read .mat files (should already be a "
            "project dependency — check your environment).") from e
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)
    return loadmat(path_or_buffer)


def inspect_mat_fields(path_or_buffer) -> Dict[str, Dict]:
    """Return a summary of every top-level field in a .mat file: shape,
    dtype, and whether it's a plain numeric matrix vs. something
    scipy.io.loadmat can't cleanly hand back as numbers (a MATLAB struct,
    cell array, or char/string array — these load as numpy object/void
    arrays, and blindly forcing them to float produces confusing errors
    like "could not convert string to float: '[0E0,0E0,...]'" instead of
    a clear "this field isn't numeric" message). Use this to build a field
    picker UI, or to pre-check a field before load_mat_numeric_field().
    """
    mat = _load_mat(path_or_buffer)
    info = {}
    for key, val in mat.items():
        if key.startswith('__'):
            continue
        arr = np.asarray(val)
        is_numeric = np.issubdtype(arr.dtype, np.number)
        info[key] = {
            'shape': arr.shape,
            'dtype': str(arr.dtype),
            'ndim': arr.ndim,
            'is_numeric': bool(is_numeric),
        }
    return info


def load_mat_numeric_field(path_or_buffer, field: str) -> np.ndarray:
    """Extract one field from a .mat file as a plain float ndarray, with a
    clear, specific error if the field isn't actually a plain numeric
    matrix (MATLAB structs/cells/strings all load via scipy as
    object/void dtype arrays that silently produce garbage — or a
    late, confusing ValueError deep inside pandas/sklearn/shap — if you
    just cast them to float without checking first).
    """
    mat = _load_mat(path_or_buffer)
    available = [k for k in mat.keys() if not k.startswith('__')]
    if field not in mat:
        raise ValueError(f"'{field}' not found in .mat file. Available keys: {available}")

    arr = np.asarray(mat[field])
    if not np.issubdtype(arr.dtype, np.number):
        sample = repr(arr.flat[0])[:80] if arr.size else "(empty)"
        raise ValueError(
            f"'{field}' isn't a plain numeric matrix (dtype={arr.dtype}, shape={arr.shape}, "
            f"first element looks like {sample}) — it's probably a MATLAB struct, cell array, "
            f"or string field that scipy.io.loadmat can't convert directly. Available keys: "
            f"{available}. If '{field}' is meant to hold numbers, open the file in MATLAB/Octave "
            f"and check how it's actually stored.")
    return arr.astype(float)


def load_ninapro_mat(
    path_or_buffer,
    fs: int,
    label_field: str = "restimulus",
    emg_field: str = "emg",
    window_ms: int = 200,
    overlap: float = 0.5,
    threshold_multiplier: float = 1.0,
    subject_id: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load a NinaPro .mat file (Atzori et al. 2014, ninaweb.hevs.ch).

    fs is REQUIRED — NinaPro's sub-databases use different hardware at
    different rates and this isn't reliably stored in the .mat file
    itself. Per the NinaPro papers and official documentation:
      DB1: 100 Hz  (10-ch Otto Bock, RMS-filtered electrodes)
      DB2, DB3: 2000 Hz  (12-ch Delsys Trigno)
      DB4: 2000 Hz  (12-ch Cometa)
      DB5: 200 Hz  (16-ch, two Myo armbands)
    Confirm against https://ninapro.hevs.ch/ for your specific database
    (especially DB6-DB10) before trusting this — don't assume.

    label_field defaults to 'restimulus' rather than 'stimulus': NinaPro's
    own documentation describes restimulus as the a-posteriori relabeled,
    more accurate version of the movement label (Gijsberts et al. 2014).
    Falls back to 'stimulus' if 'restimulus' isn't present in the file.

    Each output window's label is the majority vote of the per-sample
    labels inside that window. threshold_multiplier plays the same role
    as EMGConfig.feature_threshold_multiplier (see its docstring).
    """
    mat = _load_mat(path_or_buffer)
    available = [k for k in mat.keys() if not k.startswith('__')]

    if emg_field not in mat:
        raise ValueError(
            f"'{emg_field}' not found in .mat file. Available keys: {available}")
    raw_signal = load_mat_numeric_field(path_or_buffer, emg_field)

    label_source = label_field
    if label_source not in mat:
        fallback = 'stimulus' if label_field == 'restimulus' else 'restimulus'
        if fallback in mat:
            label_source = fallback
        else:
            raise ValueError(
                f"Neither '{label_field}' nor '{fallback}' found in .mat file. "
                f"Available keys: {available}")
    labels_per_sample = load_mat_numeric_field(path_or_buffer, label_source).flatten()

    result = _window_and_extract_with_majority_label(
        raw_signal, labels_per_sample, fs=fs, window_ms=window_ms,
        overlap=overlap, threshold_multiplier=threshold_multiplier,
        subject_id=subject_id, channel=0,
    )
    result['label_field_used'] = label_source
    return result


def load_features_csv(
    path: Union[str, Path],
    label_col: str = "label",
    group_col: Optional[str] = "subject",
) -> Dict[str, np.ndarray]:
    """Load a table of PRE-EXTRACTED features (one row per window).

    Expected columns: the 7 time-domain feature names in TIME_FEATURES
    (MAV, RMS, ZCR, WL, SSC, WAMP, MYOP) plus `label_col` and, if you have
    multiple subjects/sessions and want LOSO evaluation, `group_col`.

    Returns a dict with 'features' (list of per-window feature dicts, the
    same shape generate_synthetic_emg_dataset() produces), 'labels', and
    'groups' (None if group_col is missing or not provided).
    """
    df = robust_read_table(path)
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' column not found in {path}. "
                          f"Available columns: {list(df.columns)}")

    feature_cols = [c for c in TIME_FEATURES if c in df.columns]
    missing = [c for c in TIME_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected feature columns {missing}. "
            f"If your feature names differ, extract raw signal with "
            f"load_raw_emg_and_extract() instead so column naming can't drift.")

    features = df[feature_cols].to_dict(orient="records")
    labels = df[label_col].to_numpy()
    groups = df[group_col].to_numpy() if (group_col and group_col in df.columns) else None

    return {'features': features, 'labels': labels, 'groups': groups}


def load_raw_emg_and_extract(
    raw_signal: np.ndarray,
    labels: np.ndarray,
    label_times: np.ndarray,
    fs: int,
    window_ms: int = 100,
    overlap: float = 0.5,
    channel: int = 0,
    threshold: float = 0.0,
    subject_id: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Window + extract features from a RAW (n_samples, n_channels) signal
    with a separate, sparser label timeline (labels[i] is valid from
    label_times[i] until the next entry) — the format most raw EMG
    datasets ship in (a continuous recording plus a stimulus/rep vector).

    threshold should be the noise-relative threshold used elsewhere in
    this app (EMGEngine computes it as feature_threshold_multiplier x the
    noise floor of the differenced signal — see engine.py process()); if
    you're calling this directly, pass 0.0 only if you understand that
    WAMP/MYOP/ZCR/SSC will then be degenerate (see EMGConfig docstring for
    feature_threshold_multiplier).
    """
    if raw_signal.ndim == 1:
        raw_signal = raw_signal.reshape(-1, 1)
    window_size = int(window_ms * fs / 1000)

    all_features = extract_features_stream(
        raw_signal, fs, window_size=window_size, overlap=overlap,
        include_freq=False, threshold=threshold,
    )
    ch_features = all_features[channel]

    step = max(1, int(window_size * (1 - overlap)))
    window_times = np.array([i * step / fs for i in range(len(ch_features))])

    window_labels = []
    for wt in window_times:
        mask = label_times <= wt
        window_labels.append(labels[mask][-1] if mask.any() else labels[0])

    groups = None
    if subject_id is not None:
        groups = np.array([subject_id] * len(ch_features))

    return {
        'features': ch_features,
        'labels': np.array(window_labels),
        'groups': groups,
    }


def concatenate_subjects(subject_datasets: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Merge several load_raw_emg_and_extract()/load_features_csv() outputs
    (one per subject) into a single dataset ready for
    EMGClassifier.fit(X, y, groups=...) with LOSO cross-validation across
    real subjects, not synthetic ones.
    """
    features, labels, groups = [], [], []
    for i, ds in enumerate(subject_datasets):
        features.extend(ds['features'])
        labels.extend(ds['labels'])
        g = ds.get('groups')
        if g is None:
            g = [f"subject_{i}"] * len(ds['features'])
        groups.extend(list(g))
    return {
        'features': features,
        'labels': np.array(labels),
        'groups': np.array(groups),
    }
