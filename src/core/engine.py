"""
engine.py - EMG Signal Processing Engine
MyoControl Suite v0.3

The main engine orchestrates: filter design → preprocess → feature extraction →
statistics → quality assessment. Fixes the v0.2 bugs (hardcoded fatigue_index,
hardcoded artifact_detected) and provides a cleaner public API.

Public API:
    engine = EMGEngine(EMGConfig(...))
    result = engine.process(signal_2d)
"""
import numpy as np
import logging
import time
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import asdict

from scipy import signal as scipy_signal

from .config import EMGConfig
from .features import (
    extract_features_stream,
    fatigue_index,
    mnf_slope,
    estimate_noise_floor,
    TIME_FEATURES, FREQ_FEATURES,
)

logger = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class EMGEngine:
    """
    Main EMG processing engine.

    Usage:
        config = EMGConfig(sampling_rate=2000)
        engine = EMGEngine(config)
        result = engine.process(raw_emg)  # shape (N, C)
    """

    def __init__(self, config: EMGConfig):
        self.config = config
        self.config.validate()
        self.filters = self._design_filters()
        logger.info(
            f"EMGEngine initialized: fs={config.sampling_rate}Hz, "
            f"filter={config.filter_type} "
            f"({config.cutoff_low}-{config.cutoff_high}Hz, order={config.filter_order})"
        )

    # --------------------------------------------------------
    # Filter design
    # --------------------------------------------------------
    def _design_filters(self) -> Dict:
        nyquist = self.config.sampling_rate / 2
        Wn = [
            self.config.cutoff_low / nyquist,
            self.config.cutoff_high / nyquist,
        ]

        ft = self.config.filter_type
        try:
            if ft == 'butterworth':
                b_band, a_band = scipy_signal.butter(
                    self.config.filter_order, Wn, btype='band')
            elif ft == 'chebyshev':
                b_band, a_band = scipy_signal.cheby1(
                    self.config.filter_order, 0.5, Wn, btype='band')
            elif ft == 'bessel':
                b_band, a_band = scipy_signal.bessel(
                    self.config.filter_order, Wn, btype='band')
            elif ft == 'elliptic':
                b_band, a_band = scipy_signal.ellip(
                    self.config.filter_order, 0.5, 40, Wn, btype='band')
            else:
                raise ValueError(f"Unknown filter type: {ft}")
        except Exception as e:
            logger.error(f"Filter design failed: {e}")
            raise

        b_notch, a_notch = scipy_signal.iirnotch(
            self.config.notch_freq / nyquist,
            self.config.notch_quality,
        )
        return {'bandpass': (b_band, a_band), 'notch': (b_notch, a_notch)}

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------
    def preprocess(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Apply bandpass + notch filter. Always returns 2D (samples, channels).
        NaN/Inf values are replaced with 0 to prevent propagation.
        """
        if raw_signal.ndim == 1:
            raw_signal = raw_signal.reshape(-1, 1)
        elif raw_signal.ndim != 2:
            raise ValueError(
                f"Expected 1D or 2D array, got {raw_signal.ndim}D")

        # Guard against NaN/Inf (replaces with 0)
        if not np.all(np.isfinite(raw_signal)):
            n_bad = int(np.sum(~np.isfinite(raw_signal)))
            logger.warning(
                f"Signal contains {n_bad} non-finite values "
                f"(NaN/Inf). Replacing with 0.")
            raw_signal = np.nan_to_num(raw_signal, nan=0.0,
                                        posinf=0.0, neginf=0.0)

        n_samples, n_channels = raw_signal.shape
        if n_samples < self.config.window_size:
            raise ValueError(
                f"Signal length {n_samples} < window size {self.config.window_size}")

        b_band, a_band = self.filters['bandpass']
        b_notch, a_notch = self.filters['notch']

        filtered = np.zeros_like(raw_signal, dtype=np.float64)
        for ch in range(n_channels):
            f = scipy_signal.filtfilt(b_band, a_band, raw_signal[:, ch])
            f = scipy_signal.filtfilt(b_notch, a_notch, f)
            filtered[:, ch] = f

        return filtered

    # --------------------------------------------------------
    # Noise estimation
    # --------------------------------------------------------
    def _estimate_noise_floor(self, signal_segment: np.ndarray) -> float:
        """Estimate noise floor from a 1D segment (delegates to the shared
        implementation in features.py — see estimate_noise_floor())."""
        return estimate_noise_floor(
            signal_segment,
            sampling_rate=self.config.sampling_rate,
            method=self.config.noise_estimation_method,
            percentile=self.config.noise_percentile,
            manual_value=self.config.manual_noise_floor,
        )

    # --------------------------------------------------------
    # Artifact detection (FIXED — was hardcoded False in v0.2)
    # --------------------------------------------------------
    def _detect_artifacts(self, signal_2d: np.ndarray,
                          noise_floor: float) -> Dict:
        """
        Detect movement artifacts: samples > threshold × noise_floor.
        Returns artifact_detected flag + indices + percentage.
        """
        try:
            # Per-channel peak-to-noise ratio
            std_per_ch = np.std(signal_2d, axis=0)
            if np.any(noise_floor > 0):
                ratios = std_per_ch / max(noise_floor, 1e-9)
            else:
                ratios = std_per_ch / 1e-9

            # An artifact is any sample exceeding threshold × std
            threshold = self.config.artifact_threshold * np.std(signal_2d)
            artifact_mask = np.any(np.abs(signal_2d) > threshold, axis=1)
            artifact_pct = float(np.mean(artifact_mask) * 100)

            return {
                'artifact_detected': bool(artifact_pct > 1.0),
                'artifact_percentage': artifact_pct,
                'artifact_samples': int(np.sum(artifact_mask)),
                'peak_to_noise_ratio': float(np.max(ratios)),
            }
        except Exception as e:
            logger.error(f"Artifact detection failed: {e}")
            return {
                'artifact_detected': False,
                'artifact_percentage': 0.0,
                'artifact_samples': 0,
                'peak_to_noise_ratio': 0.0,
            }

    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------
    def _compute_snr(self, signal_1d: np.ndarray, noise_floor: float) -> float:
        if noise_floor <= 0:
            return 0.0
        std = float(np.std(signal_1d))
        if std <= 0:
            return 0.0
        return float(20 * np.log10(std / noise_floor))

    # --------------------------------------------------------
    # Main process method
    # --------------------------------------------------------
    def process(self,
                raw_signal: np.ndarray,
                selected_channel: int = 0,
                measure_time: bool = False,
                compute_freq_features: Optional[bool] = None) -> Dict:
        """
        Process EMG signal and return full analysis.

        Parameters
        ----------
        raw_signal : (N,) or (N, C) array
        selected_channel : which channel to highlight in plots/SNR
        measure_time : include processing time in output
        compute_freq_features : override config setting
        """
        start_time = time.perf_counter() if measure_time else None

        if compute_freq_features is None:
            compute_freq_features = self.config.compute_frequency_features

        # Ensure 2D
        if raw_signal.ndim == 1:
            raw_signal = raw_signal.reshape(-1, 1)
        elif raw_signal.ndim != 2:
            raise ValueError(
                f"raw_signal must be 1D or 2D, got {raw_signal.ndim}D")

        n_samples, n_channels = raw_signal.shape

        # Memory check
        if HAS_PSUTIL:
            mem_est = self._estimate_memory(n_samples, n_channels)
            avail = psutil.virtual_memory().available / (1024 ** 2)
            if mem_est > 0.5 * avail:
                logger.warning(
                    f"Estimated memory {mem_est:.1f} MB > 50% of available "
                    f"{avail:.1f} MB. Consider chunking."
                )

        # Filter
        filtered = self.preprocess(raw_signal)

        # Per-channel noise floor + SNR + artifact detection
        ch_data = filtered[:, selected_channel] if n_channels > 1 else filtered[:, 0]
        noise_floor = self._estimate_noise_floor(ch_data)
        snr_db = self._compute_snr(ch_data, noise_floor)
        artifact_info = self._detect_artifacts(filtered, noise_floor)

        # Feature extraction. WAMP/MYOP/ZCR/SSC need a noise-relative
        # crossing threshold (see EMGConfig.feature_threshold_multiplier) —
        # without it they saturate to a constant value regardless of muscle
        # activity, which makes them useless for classification. WAMP/SSC
        # compare against the sample-to-sample derivative, so the threshold
        # is calibrated on that same scale (not the amplitude noise floor).
        diff_noise_floor = self._estimate_noise_floor(np.abs(np.diff(ch_data)))
        feature_threshold = self.config.feature_threshold_multiplier * diff_noise_floor
        all_features = extract_features_stream(
            filtered, self.config.sampling_rate,
            window_size=self.config.window_size,
            overlap=self.config.overlap,
            include_freq=compute_freq_features,
            psd_method=self.config.psd_method,
            psd_nperseg=self.config.psd_nperseg,
            threshold=feature_threshold,
        )

        # Build timestamps
        step = max(1, int(self.config.window_size * (1 - self.config.overlap)))
        n_windows = max(1, len(all_features[0]) if all_features else 0)
        timestamps = [i * step / self.config.sampling_rate
                      for i in range(n_windows)]

        # Summary stats per channel (WITH REAL fatigue index)
        summary = {}
        fs_features = self.config.sampling_rate / step  # windows per second
        for ch in range(n_channels):
            if ch >= len(all_features) or not all_features[ch]:
                continue
            ch_feats = all_features[ch]
            mav_vals = np.array([f['MAV'] for f in ch_feats])
            rms_vals = np.array([f['RMS'] for f in ch_feats])

            # Real fatigue index (was hardcoded 0.0 in v0.2)
            fi = fatigue_index(rms_vals, fs_features)

            ch_summary = {
                'mean_activation': float(np.mean(mav_vals)) if len(mav_vals) else 0.0,
                'peak_activation': float(np.max(rms_vals)) if len(rms_vals) else 0.0,
                'std_activation': float(np.std(rms_vals)) if len(rms_vals) else 0.0,
                'fatigue_index': fi,
                'n_windows': len(ch_feats),
            }

            if compute_freq_features:
                mnf_vals = np.array([f['MNF'] for f in ch_feats])
                ch_summary['mnf_slope'] = mnf_slope(mnf_vals, fs_features)
                ch_summary['mean_mnf'] = float(np.mean(mnf_vals)) if len(mnf_vals) else 0.0

            summary[f'channel_{ch}'] = ch_summary

        # Activity level classification
        activity_level = self._classify_activity(mav_vals, snr_db)

        # Build output
        output = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'engine_version': '0.3.0',
                'sampling_rate': self.config.sampling_rate,
                'window_size': self.config.window_size,
                'overlap': self.config.overlap,
                'filter_config': self.config.to_dict(),
                'n_channels': n_channels,
                'selected_channel': selected_channel,
                'n_samples': n_samples,
                'duration_seconds': n_samples / self.config.sampling_rate,
            },
            'signal_quality': {
                'estimated_noise_floor': float(noise_floor),
                'mean_snr_db': float(snr_db),
                'snr_quality': self._snr_label(snr_db),
                **artifact_info,
            },
            'clinical_interpretation': {
                'muscle_activity': activity_level,
                'activation_pattern': self._activation_pattern(rms_vals),
            },
            'time_series': {
                'timestamps': timestamps,
                'features': all_features,
            },
            'summary_statistics': summary,
        }

        if measure_time:
            output['benchmark'] = {
                'processing_time_ms': (time.perf_counter() - start_time) * 1000,
            }

        return output

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _estimate_memory(self, n_samples: int, n_channels: int) -> float:
        raw_mb = n_samples * n_channels * 8 / (1024 ** 2)
        filt_mb = raw_mb
        n_windows = n_samples // self.config.window_size
        n_feats = len(TIME_FEATURES) + len(FREQ_FEATURES)
        feat_mb = n_windows * n_channels * n_feats * 8 / (1024 ** 2)
        return raw_mb + filt_mb + feat_mb

    def _snr_label(self, snr_db: float) -> str:
        cfg = self.config
        if snr_db < cfg.snr_threshold_poor:
            return 'poor'
        if snr_db < cfg.snr_threshold_fair:
            return 'fair'
        if snr_db < cfg.snr_threshold_good:
            return 'good'
        return 'excellent'

    @staticmethod
    def _classify_activity(mav_vals: np.ndarray, snr_db: float) -> str:
        if len(mav_vals) == 0:
            return 'unknown'
        mean_mav = float(np.mean(mav_vals))
        if mean_mav < 0.01 or snr_db < 3:
            return 'very_low_resting'
        if mean_mav < 0.05:
            return 'low_light_contraction'
        if mean_mav < 0.2:
            return 'moderate_moderate_contraction'
        if mean_mav < 0.5:
            return 'high_strong_contraction'
        return 'very_high_maximal_contraction'

    @staticmethod
    def _activation_pattern(rms_vals: np.ndarray) -> str:
        if len(rms_vals) < 5:
            return 'insufficient_data'
        # Check if signal is increasing, decreasing, or stable
        x = np.arange(len(rms_vals))
        slope, _ = np.polyfit(x, rms_vals, 1)
        rms_range = float(np.max(rms_vals) - np.min(rms_vals))
        if rms_range < 0.005:
            return 'constant_isometric'
        if slope > 0.001:
            return 'increasing_ramp'
        if slope < -0.001:
            return 'decreasing_fatiguing'
        return 'variable_dynamic'
