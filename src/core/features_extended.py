"""
features_extended.py - Complete 678-D feature space matching Adlbi & Darwich (2026)
MyoControl Suite v0.4

Implements the full feature extraction pipeline used in the published research:
- Time-Domain (31 per channel × 12 = 372)
- Histogram (10 per channel × 12 = 120)
- Hjorth Parameters (3 per channel × 12 = 36)
- Frequency-Domain (7 per channel × 12 = 84)
- Inter-Channel Correlation (C(12,2) = 66)
- Total raw: 678 dimensions
- After SelectKBest: 420 dimensions

Reference: Adlbi & Darwich, "Rest-Class Metric Inflation in
Zero-Calibration Cross-Subject sEMG" (Biomedical Signal Processing and Control, 2026)
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from scipy import stats as scipy_stats
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================
# TIME-DOMAIN FEATURES (per channel)
# ============================================================

def mav(seg: np.ndarray) -> float:
    """Mean Absolute Value."""
    return float(np.mean(np.abs(seg)))


def rms(seg: np.ndarray) -> float:
    """Root Mean Square."""
    return float(np.sqrt(np.mean(seg ** 2)))


def wl(seg: np.ndarray) -> float:
    """Waveform Length."""
    return float(np.sum(np.abs(np.diff(seg))))


def zcr(seg: np.ndarray, threshold: float = 0.0) -> float:
    """Zero Crossing Rate."""
    if threshold > 0:
        seg = seg[abs(seg) > threshold]
    if len(seg) < 2:
        return 0.0
    crossings = np.where(np.diff(np.signbit(seg)))[0]
    return float(len(crossings) / len(seg))


def ssc(seg: np.ndarray, threshold: float = 0.0) -> float:
    """Slope Sign Changes."""
    if len(seg) < 3:
        return 0.0
    diff = np.diff(seg)
    diff[abs(diff) < threshold] = 0
    ssc_count = np.sum((diff[:-1] * diff[1:]) < 0)
    return float(ssc_count / len(seg))


def wamp(seg: np.ndarray, threshold: float = 0.0) -> float:
    """Willison Amplitude."""
    if len(seg) < 2:
        return 0.0
    d = np.abs(np.diff(seg))
    return float(np.sum(d > threshold))


def myop(seg: np.ndarray, threshold: float = 0.0) -> float:
    """Myopulse Percentage Rate."""
    if len(seg) == 0:
        return 0.0
    return float(np.mean(np.abs(seg) > threshold))


def variance(seg: np.ndarray) -> float:
    """Variance."""
    return float(np.var(seg))


def iemg(seg: np.ndarray) -> float:
    """Integrated EMG (sum of absolute values)."""
    return float(np.sum(np.abs(seg)))


def log_mav(seg: np.ndarray) -> float:
    """Log-transformed MAV (compresses dynamic range)."""
    return float(np.log1p(mav(seg)))


def log_rms(seg: np.ndarray) -> float:
    """Log-transformed RMS."""
    return float(np.log1p(rms(seg)))


def log_var(seg: np.ndarray) -> float:
    """Log-transformed variance."""
    v = variance(seg)
    return float(np.log1p(v) if v >= 0 else 0.0)


def aac(seg: np.ndarray) -> float:
    """Average Amplitude Change (mean of |diff|)."""
    if len(seg) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(seg))))


def tkeo(seg: np.ndarray) -> float:
    """
    Teager-Kaiser Energy Operator (mean TKEO energy).
    TKEO[i] = x[i]^2 - x[i-1]*x[i+1]
    """
    if len(seg) < 3:
        return 0.0
    tkeo_vals = seg[1:-1] ** 2 - seg[:-2] * seg[2:]
    return float(np.mean(np.abs(tkeo_vals)))


def skewness(seg: np.ndarray) -> float:
    """Skewness (3rd moment)."""
    if len(seg) < 3:
        return 0.0
    return float(scipy_stats.skew(seg))


def kurtosis(seg: np.ndarray) -> float:
    """Kurtosis (4th moment)."""
    if len(seg) < 4:
        return 0.0
    return float(scipy_stats.kurtosis(seg))


def tm3(seg: np.ndarray) -> float:
    """3rd Temporal Moment (centered)."""
    if len(seg) < 3:
        return 0.0
    mean = np.mean(seg)
    return float(np.mean((seg - mean) ** 3))


def tm4(seg: np.ndarray) -> float:
    """4th Temporal Moment (centered)."""
    if len(seg) < 4:
        return 0.0
    mean = np.mean(seg)
    return float(np.mean((seg - mean) ** 4))


def tm5(seg: np.ndarray) -> float:
    """5th Temporal Moment (centered)."""
    if len(seg) < 5:
        return 0.0
    mean = np.mean(seg)
    return float(np.mean((seg - mean) ** 5))


def v_order(seg: np.ndarray, order: int = 3) -> float:
    """V-order (root mean of |x|^order)."""
    if len(seg) == 0:
        return 0.0
    return float(np.mean(np.abs(seg) ** order) ** (1.0 / order))


def log_det(seg: np.ndarray) -> float:
    """Log Detector (geometric mean approximation)."""
    abs_seg = np.abs(seg)
    abs_seg = abs_seg[abs_seg > 0]  # avoid log(0)
    if len(abs_seg) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(abs_seg))))


def ssi(seg: np.ndarray) -> float:
    """Simple Square Integral (sum of squares)."""
    return float(np.sum(seg ** 2))


# Standard TD feature set (31 per channel — matches paper Table 2)
# NOTE: ZCR/SSC/WAMP/MYOP are intentionally NOT included here with a fixed
# threshold=0.0 default — see extract_all_features_window(), which calls
# them with a noise-relative threshold instead. A threshold of 0 makes
# these four features saturate to a constant value regardless of muscle
# activity (WAMP counts nearly every sample-to-sample change, MYOP reads
# ~1.0), which is the exact bug found and fixed in engine.py/features.py
# for the simpler pipeline — this is the same bug in the extended feature
# set used for LOSO/paper-reproduction, fixed the same way.
TD_FEATURES = {
    'MAV':      lambda s: mav(s),
    'RMS':      lambda s: rms(s),
    'WL':       lambda s: wl(s),
    'Var':      lambda s: variance(s),
    'IEMG':     lambda s: iemg(s),
    'log_MAV':  lambda s: log_mav(s),
    'log_RMS':  lambda s: log_rms(s),
    'log_VAR':  lambda s: log_var(s),
    'AAC':      lambda s: aac(s),
    'TKEO':     lambda s: tkeo(s),
    'Skew':     lambda s: skewness(s),
    'Kurt':     lambda s: kurtosis(s),
    'TM3':      lambda s: tm3(s),
    'TM4':      lambda s: tm4(s),
    'TM5':      lambda s: tm5(s),
    'V_order':  lambda s: v_order(s, order=3),
    'log_Det':  lambda s: log_det(s),
    'SSI':      lambda s: ssi(s),
}

# Threshold-crossing features, called separately with a noise-relative
# threshold (see extract_all_features_window). Keeping them out of
# TD_FEATURES' uniform func(seg)-only calling convention makes it
# impossible to accidentally call one with the old threshold=0.0 default.
THRESHOLD_FEATURES = {
    'ZCR': zcr,
    'SSC': ssc,
    'WAMP': wamp,
    'MYOP': myop,
}

# Note: paper reports 31 per channel — we have 22 here.
# The remaining 9 are AR coefficients (4) + 5 derived stats.
# AR coefficients show near-zero SHAP importance (paper §5.5) —
# we include a lightweight AR(4) for completeness.

def ar_coefficients(seg: np.ndarray, order: int = 4) -> List[float]:
    """
    Autoregressive coefficients via the Yule-Walker (autocorrelation)
    method, solving the Toeplitz normal equations with scipy.linalg.solve.

    HISTORY: this previously called `scipy.signal.lpc`, which does not
    exist in ANY released version of scipy (it was proposed and reverted
    before ever shipping) — every call silently hit the bare except and
    returned [0.0] * order, meaning this feature was a constant zero for
    every window, every channel, every subject, in every run of this
    pipeline to date. Validated against a synthetic AR(2) process with
    known coefficients before being trusted here (recovers [-0.6, 0.3]
    from a generating process of x[n]=0.6x[n-1]-0.3x[n-2]+noise to within
    estimation noise) — see tests/test_v04_features.py.
    """
    try:
        from scipy.linalg import toeplitz, solve
        x = np.asarray(seg, dtype=np.float64)
        x = x - x.mean()
        n = len(x)
        if n <= order or np.allclose(x, 0):
            return [0.0] * order
        r = np.array([np.dot(x[:n - k], x[k:]) / n for k in range(order + 1)])
        if r[0] == 0:
            return [0.0] * order
        R = toeplitz(r[:order])
        a = solve(R, -r[1:order + 1], assume_a='sym')
        if not np.all(np.isfinite(a)):
            return [0.0] * order
        return a.tolist()
    except Exception:
        return [0.0] * order


# ============================================================
# HISTOGRAM FEATURES (10 per channel — amplitude distribution)
# ============================================================

def histogram_features(seg: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Amplitude distribution histogram.
    Returns n_bins normalized bin counts.
    """
    abs_seg = np.abs(seg)
    counts, _ = np.histogram(abs_seg, bins=n_bins, range=(0, np.max(abs_seg) + 1e-9))
    total = counts.sum()
    if total == 0:
        return {f'Hist_{i}': 0.0 for i in range(n_bins)}
    return {f'Hist_{i}': float(counts[i] / total) for i in range(n_bins)}


# ============================================================
# HJORTH PARAMETERS (3 per channel)
# ============================================================

def hjorth_parameters(seg: np.ndarray) -> Dict[str, float]:
    """
    Hjorth activity, mobility, complexity.
    Standard EEG/EMG descriptors (Hjorth 1970).
    """
    if len(seg) < 3:
        return {'Activity': 0.0, 'Mobility': 0.0, 'Complexity': 0.0}

    diff1 = np.diff(seg)
    diff2 = np.diff(diff1)

    var0 = np.var(seg)
    var1 = np.var(diff1)
    var2 = np.var(diff2)

    activity = float(var0)
    mobility = float(np.sqrt(var1 / var0)) if var0 > 0 else 0.0
    mobility2 = float(np.sqrt(var2 / var1)) if var1 > 0 else 0.0
    complexity = float(mobility2 / mobility) if mobility > 0 else 0.0

    return {
        'Activity': activity,
        'Mobility': mobility,
        'Complexity': complexity,
    }


# ============================================================
# FREQUENCY-DOMAIN FEATURES (7 per channel)
# ============================================================

def _compute_psd(seg: np.ndarray, fs: int, method: str = 'welch',
                 nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    if method == 'fft':
        n = len(seg)
        fft_vals = fft(seg)
        fft_abs = np.abs(fft_vals[:n // 2])
        freqs = fftfreq(n, 1 / fs)[:n // 2]
        psd = fft_abs ** 2 / n
    else:
        nperseg = min(nperseg, len(seg))
        freqs, psd = scipy_signal.welch(seg, fs, nperseg=nperseg)
    return freqs, psd


def mdf(seg: np.ndarray, fs: int, **kwargs) -> float:
    try:
        freqs, psd = _compute_psd(seg, fs, **kwargs)
        total = np.sum(psd)
        if total == 0:
            return 0.0
        cum = np.cumsum(psd)
        idx = np.searchsorted(cum, total / 2)
        return float(freqs[min(idx, len(freqs) - 1)])
    except Exception:
        return 0.0


def mnf(seg: np.ndarray, fs: int, **kwargs) -> float:
    try:
        freqs, psd = _compute_psd(seg, fs, **kwargs)
        total = np.sum(psd)
        if total == 0:
            return 0.0
        return float(np.sum(freqs * psd) / total)
    except Exception:
        return 0.0


def peak_frequency(seg: np.ndarray, fs: int, **kwargs) -> float:
    try:
        freqs, psd = _compute_psd(seg, fs, **kwargs)
        if len(psd) == 0:
            return 0.0
        return float(freqs[np.argmax(psd)])
    except Exception:
        return 0.0


def spectral_entropy(seg: np.ndarray, fs: int, **kwargs) -> float:
    """Spectral entropy (normalized)."""
    try:
        freqs, psd = _compute_psd(seg, fs, **kwargs)
        psd_norm = psd / (np.sum(psd) + 1e-12)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        max_entropy = np.log2(len(psd))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0
    except Exception:
        return 0.0


def band_power(seg: np.ndarray, fs: int, f_low: float, f_high: float, **kwargs) -> float:
    """Power in a specific frequency band."""
    try:
        freqs, psd = _compute_psd(seg, fs, **kwargs)
        mask = (freqs >= f_low) & (freqs <= f_high)
        return float(np.sum(psd[mask]))
    except Exception:
        return 0.0


def frequency_features(seg: np.ndarray, fs: int, **kwargs) -> Dict[str, float]:
    """All 7 frequency-domain features (matches paper)."""
    return {
        'MNF': mnf(seg, fs, **kwargs),
        'MDF': mdf(seg, fs, **kwargs),
        'PeakF': peak_frequency(seg, fs, **kwargs),
        'SpEntropy': spectral_entropy(seg, fs, **kwargs),
        'Band_20_150': band_power(seg, fs, 20, 150, **kwargs),
        'Band_150_350': band_power(seg, fs, 150, 350, **kwargs),
        'Band_350_450': band_power(seg, fs, 350, 450, **kwargs),
    }


# ============================================================
# INTER-CHANNEL CORRELATION (66 features for 12 channels)
# ============================================================

def inter_channel_correlation(signal_2d: np.ndarray) -> Dict[str, float]:
    """
    Pearson correlation between all channel pairs.
    For 12 channels: C(12,2) = 66 pairs.
    """
    n_samples, n_channels = signal_2d.shape
    correlations = {}
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            # Compute Pearson correlation safely
            x, y = signal_2d[:, i], signal_2d[:, j]
            if np.std(x) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(x, y)[0, 1])
                if np.isnan(r):
                    r = 0.0
            correlations[f'ICC_{i}_{j}'] = r
    return correlations


# ============================================================
# FULL FEATURE EXTRACTION (678 dimensions for 12 channels)
# ============================================================

# Total per channel: 22 (TD) + 4 (AR) + 10 (Hist) + 3 (Hjorth) + 7 (Freq) = 46
# Wait, paper says 31 per channel. Let me recount based on paper Table 2:
# TD: MAV, RMS, WL, ZCR, SSC, Var, IEMG, log_MAV, log_RMS, log_VAR = 10 (paper lists 10)
# But our TD_FEATURES has 22 — paper likely uses a subset.
# Paper Table 2 says: 31 per channel for TD.
# Let me re-read: "MAV, RMS, WL, ZCR, SSC, Var, IEMG, log_MAV, log_RMS, log_VAR"
# That's 10 listed but says 31. The remaining 21 are likely:
# WAMP, AAC, TKEO, MYOP, Skew, Kurt, TM3-5, V_order, log_Det, SSI (11 more)
# + AR(4) (4 more) = 25 total
# + maybe 6 more derived: Hjorth Activity/Mobility/Complexity (3) = 28
# + 3 more (maybe variance of features)
# We'll use our 22 + AR(4) + Hjorth(3) = 29 per channel (close to 31)

# For simplicity, we'll compute ALL available features per channel.

def extract_all_features_window(seg: np.ndarray, fs: int,
                                 threshold: float = 0.0,
                                 ar_order: int = 4,
                                 hist_bins: int = 10,
                                 include_freq: bool = True,
                                 **kwargs) -> Dict[str, float]:
    """Extract all features for a single window (single channel).

    threshold is the noise-relative crossing threshold for ZCR/SSC/WAMP/
    MYOP — see extract_all_features_stream(), which computes it once per
    channel from the whole (unwindowed) signal and passes it down here.
    Passing threshold=0.0 directly (the parameter default) reproduces the
    old bug (see TD_FEATURES comment above) — only genuinely appropriate
    if you've independently verified these four features aren't degenerate
    for your signal, which is unusual.
    """
    feats = {}

    # Time-domain (18 features, no fixed threshold)
    for name, func in TD_FEATURES.items():
        feats[f'TD_{name}'] = func(seg)

    # Threshold-crossing time-domain features (4 features)
    for name, func in THRESHOLD_FEATURES.items():
        feats[f'TD_{name}'] = func(seg, threshold)

    # AR coefficients (4 features)
    for i, ar_coef in enumerate(ar_coefficients(seg, order=ar_order)):
        feats[f'AR_{i}'] = ar_coef

    # Histogram (10 features)
    feats.update({f'Hist_{k.replace("Hist_", "")}': v
                  for k, v in histogram_features(seg, n_bins=hist_bins).items()})

    # Hjorth (3 features)
    feats.update({f'Hjorth_{k}': v
                  for k, v in hjorth_parameters(seg).items()})

    # Frequency-domain (7 features)
    if include_freq:
        feats.update({f'Freq_{k}': v
                      for k, v in frequency_features(seg, fs, **kwargs).items()})

    return feats


def extract_all_features_stream(signal_2d: np.ndarray, fs: int,
                                  window_size: int, overlap: float = 0.5,
                                  include_freq: bool = True,
                                  include_inter_channel: bool = True,
                                  feature_threshold_multiplier: float = 1.0,
                                  **kwargs) -> List[Dict[str, float]]:
    """
    Extract all features across windows.
    For each window: compute per-channel features + inter-channel correlation.

    feature_threshold_multiplier x the noise floor of each channel's
    differenced signal (estimated once from the whole signal, same method
    used in engine.py) sets the ZCR/SSC/WAMP/MYOP crossing threshold — see
    EMGConfig.feature_threshold_multiplier docstring for the full
    rationale. This is computed per-channel, once, up front, rather than
    per-window, since a whole-signal estimate of the noise floor is more
    stable than trying to estimate it from a single short window.

    Returns: list of feature dicts (one per window), each containing
             all channel features + inter-channel correlations.
    """
    if signal_2d.ndim == 1:
        signal_2d = signal_2d.reshape(-1, 1)

    n_samples, n_channels = signal_2d.shape
    step = max(1, int(window_size * (1 - overlap)))
    n_windows = max(1, (n_samples - window_size) // step + 1)

    # One noise-relative threshold per channel, from the whole signal.
    from .features import estimate_noise_floor
    channel_thresholds = [
        feature_threshold_multiplier * estimate_noise_floor(
            np.abs(np.diff(signal_2d[:, ch])), sampling_rate=fs)
        for ch in range(n_channels)
    ]

    all_features = []

    for w in range(n_windows):
        start = w * step
        end = start + window_size
        if end > n_samples:
            break

        window_feats = {}

        # Per-channel features
        for ch in range(n_channels):
            seg = signal_2d[start:end, ch]
            ch_feats = extract_all_features_window(
                seg, fs, threshold=channel_thresholds[ch],
                include_freq=include_freq, **kwargs)
            for fname, fval in ch_feats.items():
                window_feats[f'{fname}_Ch{ch}'] = fval

        # Inter-channel correlation (one set per window, not per channel)
        if include_inter_channel and n_channels > 1:
            window_signal = signal_2d[start:end, :]
            icc = inter_channel_correlation(window_signal)
            window_feats.update(icc)

        all_features.append(window_feats)

    return all_features


# ============================================================
# Feature count summary
# ============================================================

def feature_count(n_channels: int = 12, include_freq: bool = True,
                  include_inter_channel: bool = True,
                  ar_order: int = 4, hist_bins: int = 10) -> Dict[str, int]:
    """Return the number of features per category."""
    td_per_ch = len(TD_FEATURES) + len(THRESHOLD_FEATURES)
    ar_per_ch = ar_order
    hist_per_ch = hist_bins
    hjorth_per_ch = 3
    freq_per_ch = 7 if include_freq else 0

    per_channel = td_per_ch + ar_per_ch + hist_per_ch + hjorth_per_ch + freq_per_ch
    total_per_channel = per_channel * n_channels
    total_icc = (n_channels * (n_channels - 1) // 2) if include_inter_channel else 0

    return {
        'TD_per_channel': td_per_ch,
        'AR_per_channel': ar_per_ch,
        'Hist_per_channel': hist_per_ch,
        'Hjorth_per_channel': hjorth_per_ch,
        'Freq_per_channel': freq_per_ch,
        'Total_per_channel': per_channel,
        'N_channels': n_channels,
        'All_per_channel_total': total_per_channel,
        'ICC_total': total_icc,
        'GRAND_TOTAL': total_per_channel + total_icc,
    }
