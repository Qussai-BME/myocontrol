"""
features.py - EMG Feature Extraction (Time + Frequency Domain)
MyoControl Suite v0.3

Implements the 7 classic Hudgins (1993) + Phinyomark (2012) features
plus extended frequency-domain features for fatigue analysis.

References:
- Hudgins et al. (1993) IEEE Trans BME
- Phinyomark et al. (2012) Expert Systems with Applications
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def estimate_noise_floor(
    signal_segment: np.ndarray,
    sampling_rate: int,
    method: str = 'percentile',
    percentile: float = 5.0,
    manual_value: Optional[float] = None,
) -> float:
    """Estimate the noise floor of a 1D signal from the low-percentile RMS
    of sliding sub-windows — i.e. an estimate of the "quiet" baseline
    level, used to set noise-relative thresholds elsewhere (SNR, artifact
    detection, and the WAMP/MYOP/ZCR/SSC crossing threshold — see
    EMGConfig.feature_threshold_multiplier for why those need this rather
    than a threshold of 0). This is the single implementation shared by
    EMGEngine and dataset_loader.py; don't reimplement it inline.
    """
    try:
        if method == 'manual' and manual_value is not None:
            return float(manual_value)

        window_len = int(0.2 * sampling_rate)
        if window_len < 10:
            window_len = min(10, len(signal_segment) // 4)
        if len(signal_segment) < window_len:
            return float(np.std(signal_segment))

        rms_vals = []
        for i in range(0, len(signal_segment) - window_len,
                       max(1, window_len // 2)):
            seg = signal_segment[i:i + window_len]
            rms_vals.append(np.sqrt(np.mean(seg ** 2)))

        if not rms_vals:
            return float(np.std(signal_segment))

        if method == 'median':
            return float(np.median(rms_vals))
        return float(np.percentile(rms_vals, percentile))
    except Exception as e:
        logger.error(f"Noise floor estimation failed: {e}")
        return 0.01


# ============================================================
# TIME-DOMAIN FEATURES (Hudgins 5 + extended)
# ============================================================

def mav(seg: np.ndarray) -> float:
    """Mean Absolute Value."""
    return float(np.mean(np.abs(seg)))


def rms(seg: np.ndarray) -> float:
    """Root Mean Square."""
    return float(np.sqrt(np.mean(seg ** 2)))


def zcr(seg: np.ndarray, threshold: float = 0.0) -> float:
    """Zero Crossing Rate (with optional threshold)."""
    if threshold > 0:
        seg = seg[abs(seg) > threshold]
    if len(seg) < 2:
        return 0.0
    crossings = np.where(np.diff(np.signbit(seg)))[0]
    return float(len(crossings) / len(seg))


def wl(seg: np.ndarray) -> float:
    """Waveform Length (cumulative segment length)."""
    return float(np.sum(np.abs(np.diff(seg))))


def ssc(seg: np.ndarray, threshold: float = 0.0) -> float:
    """Slope Sign Changes (number of times slope changes sign)."""
    if len(seg) < 3:
        return 0.0
    diff = np.diff(seg)
    # Apply threshold
    diff[abs(diff) < threshold] = 0
    # Slope sign change: diff[i-1] * diff[i] < 0
    ssc_count = np.sum((diff[:-1] * diff[1:]) < 0)
    return float(ssc_count / len(seg))


def wamp(seg: np.ndarray, threshold: float = 0.0) -> float:
    """Willison Amplitude — number of times diff exceeds threshold."""
    if len(seg) < 2:
        return 0.0
    d = np.abs(np.diff(seg))
    return float(np.sum(d > threshold))


def myop(seg: np.ndarray, threshold: float = 0.0) -> float:
    """Myopulse Percentage Rate — fraction of time signal exceeds threshold."""
    if len(seg) == 0:
        return 0.0
    return float(np.mean(np.abs(seg) > threshold))


# ============================================================
# FREQUENCY-DOMAIN FEATURES (Phinyomark + fatigue)
# ============================================================

def _compute_psd(seg: np.ndarray, fs: int, method: str = 'welch',
                 nperseg: int = 256) -> tuple:
    """Compute Power Spectral Density."""
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


def mdf(seg: np.ndarray, fs: int, method: str = 'welch',
        nperseg: int = 256) -> float:
    """Median Frequency — frequency where cumulative power = 50%."""
    try:
        freqs, psd = _compute_psd(seg, fs, method, nperseg)
        total = np.sum(psd)
        if total == 0:
            return 0.0
        cum = np.cumsum(psd)
        idx = np.searchsorted(cum, total / 2)
        return float(freqs[min(idx, len(freqs) - 1)])
    except Exception:
        return 0.0


def mnf(seg: np.ndarray, fs: int, method: str = 'welch',
        nperseg: int = 256) -> float:
    """Mean Frequency (weighted average)."""
    try:
        freqs, psd = _compute_psd(seg, fs, method, nperseg)
        total = np.sum(psd)
        if total == 0:
            return 0.0
        return float(np.sum(freqs * psd) / total)
    except Exception:
        return 0.0


def peak_frequency(seg: np.ndarray, fs: int, method: str = 'welch',
                   nperseg: int = 256) -> float:
    """Peak frequency of PSD."""
    try:
        freqs, psd = _compute_psd(seg, fs, method, nperseg)
        if len(psd) == 0:
            return 0.0
        return float(freqs[np.argmax(psd)])
    except Exception:
        return 0.0


def spectral_moment(seg: np.ndarray, fs: int, n: int = 2,
                    method: str = 'welch', nperseg: int = 256) -> float:
    """n-th spectral moment (used for variance and skewness)."""
    try:
        freqs, psd = _compute_psd(seg, fs, method, nperseg)
        total = np.sum(psd)
        if total == 0:
            return 0.0
        return float(np.sum((freqs ** n) * psd) / total)
    except Exception:
        return 0.0


# ============================================================
# FATIGUE INDEX (real implementation, fixes hardcoded 0.0)
# ============================================================

def fatigue_index(rms_over_time: np.ndarray, fs_features: float) -> float:
    """
    Estimate muscle fatigue as the negative slope of RMS over time.

    A more negative value indicates increasing fatigue (RMS declining).
    A positive value indicates increasing activity (warm-up).

    Parameters
    ----------
    rms_over_time : array of RMS values (one per window)
    fs_features : float, feature sampling rate (windows/second)

    Returns
    -------
    float: fatigue index (negative = fatiguing, positive = activating)
    """
    if len(rms_over_time) < 3:
        return 0.0
    x = np.arange(len(rms_over_time)) / max(fs_features, 1e-6)
    slope, _ = np.polyfit(x, rms_over_time, 1)
    return float(-slope)


def mnf_slope(mnf_over_time: np.ndarray, fs_features: float) -> float:
    """
    Spectral fatigue indicator: slope of MNF over time.
    A negative slope is the gold-standard fatigue indicator
    (compression of spectrum toward lower frequencies).
    """
    if len(mnf_over_time) < 3:
        return 0.0
    x = np.arange(len(mnf_over_time)) / max(fs_features, 1e-6)
    slope, _ = np.polyfit(x, mnf_over_time, 1)
    return float(slope)


# ============================================================
# FEATURE EXTRACTION PIPELINE
# ============================================================

# Standard 7-feature set (Hudgins + Phinyomark)
TIME_FEATURES = {
    'MAV': mav,
    'RMS': rms,
    'ZCR': zcr,
    'WL':  wl,
    'SSC': ssc,
    'WAMP': wamp,
    'MYOP': myop,
}

FREQ_FEATURES = {
    'MDF': mdf,
    'MNF': mnf,
    'PKF': peak_frequency,
}


def extract_window_features(seg: np.ndarray, fs: int,
                            threshold: float = 0.0,
                            include_freq: bool = False,
                            psd_method: str = 'welch',
                            psd_nperseg: int = 256) -> Dict[str, float]:
    """
    Extract all features for a single signal window.

    Parameters
    ----------
    seg : 1D numpy array, signal window
    fs : sampling rate (Hz)
    threshold : for ZCR, SSC, WAMP, MYOP
    include_freq : whether to compute frequency-domain features
    """
    features = {}
    for name, func in TIME_FEATURES.items():
        if name in ('ZCR', 'SSC'):
            features[name] = func(seg, threshold=threshold)
        elif name in ('WAMP', 'MYOP'):
            features[name] = func(seg, threshold=threshold)
        else:
            features[name] = func(seg)

    if include_freq:
        for name, func in FREQ_FEATURES.items():
            features[name] = func(seg, fs, method=psd_method, nperseg=psd_nperseg)

    return features


def extract_features_stream(signal_2d: np.ndarray, fs: int,
                            window_size: int, overlap: float = 0.5,
                            threshold: float = 0.0,
                            include_freq: bool = False,
                            psd_method: str = 'welch',
                            psd_nperseg: int = 256) -> List[List[Dict[str, float]]]:
    """
    Extract features across windows for all channels.

    Parameters
    ----------
    signal_2d : (n_samples, n_channels) array

    Returns
    -------
    list of lists: features[ch_idx][window_idx] = {feature: value}
    """
    if signal_2d.ndim == 1:
        signal_2d = signal_2d.reshape(-1, 1)

    n_samples, n_channels = signal_2d.shape
    step = max(1, int(window_size * (1 - overlap)))
    n_windows = max(1, (n_samples - window_size) // step + 1)

    all_features = []
    for ch in range(n_channels):
        ch_feats = []
        for i in range(n_windows):
            start = i * step
            end = start + window_size
            if end > n_samples:
                break
            window = signal_2d[start:end, ch]
            feats = extract_window_features(
                window, fs, threshold=threshold,
                include_freq=include_freq,
                psd_method=psd_method, psd_nperseg=psd_nperseg,
            )
            ch_feats.append(feats)
        all_features.append(ch_feats)

    return all_features
