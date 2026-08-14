"""
realtime_inference.py - Real-time EMG inference engine
MyoControl Suite v0.5

Real-time pipeline for live EMG stream:
  Raw EMG → Filter → Windowing → Feature Extraction → ONNX Inference → Prediction

Designed for:
- < 5ms total latency (filter + features + inference)
- Sliding window with configurable overlap
- Live streaming input (callback-based)
- Optional SHAP transparency per window

Usage:
    engine = RealtimeEMGEngine(
        model_path='model.onnx',
        fs=2000, window_ms=400, overlap=0.5,
    )
    engine.start(callback=lambda pred: print(pred))
"""
import numpy as np
import logging
from typing import Optional, Callable, List, Dict
from collections import deque
import time

logger = logging.getLogger(__name__)


class RealtimeEMGEngine:
    """
    Real-time EMG classification engine with ONNX inference.

    Pipeline per window:
    1. Collect window_size samples from each channel
    2. Apply bandpass + notch filter
    3. Extract features (Numba-accelerated)
    4. Run ONNX inference
    5. Call user callback with prediction
    """

    def __init__(self,
                 model_path: str,
                 fs: int = 2000,
                 window_ms: int = 400,
                 overlap: float = 0.5,
                 n_channels: int = 8,
                 cutoff_low: float = 20.0,
                 cutoff_high: float = 450.0,
                 filter_order: int = 4,
                 notch_freq: float = 50.0,
                 smoothing_window: int = 3,
                 confidence_threshold: float = 0.5):
        """
        Parameters
        ----------
        model_path : path to ONNX model file
        fs : sampling rate (Hz)
        window_ms : window size in milliseconds
        overlap : window overlap (0-0.9)
        n_channels : number of EMG channels
        smoothing_window : majority vote over last N predictions
        confidence_threshold : minimum confidence to accept prediction
        """
        from scipy import signal as scipy_signal

        self.fs = fs
        self.window_size = int(window_ms * fs / 1000)
        self.overlap = overlap
        self.step = int(self.window_size * (1 - overlap))
        self.n_channels = n_channels
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold

        # Initialize ONNX inference
        try:
            from .onnx_export import OnnxInferenceEngine
            self.inference = OnnxInferenceEngine(model_path)
        except ImportError:
            raise ImportError("onnxruntime required: pip install onnxruntime")

        # Initialize filter coefficients
        nyquist = fs / 2
        b_band, a_band = scipy_signal.butter(
            filter_order,
            [cutoff_low / nyquist, cutoff_high / nyquist],
            btype='band',
        )
        b_notch, a_notch = scipy_signal.iirnotch(notch_freq / nyquist, 30.0)
        self.filters = {'bandpass': (b_band, a_band),
                        'notch': (b_notch, a_notch)}

        # Initialize buffer (sliding window)
        self.buffer = deque(maxlen=self.window_size + self.step)
        self.is_running = False
        self.samples_since_window = 0

        # Smoothing buffer
        self.prediction_history = deque(maxlen=smoothing_window)

        # Performance metrics
        self.total_windows = 0
        self.total_time_ms = 0.0

    def add_samples(self, samples: np.ndarray) -> Optional[Dict]:
        """
        Add new EMG samples to the buffer.

        Parameters
        ----------
        samples : (n_new, n_channels) array of new EMG samples

        Returns
        -------
        Optional[dict] with prediction if a window was processed, else None
        """
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)

        # Add to buffer
        for row in samples:
            self.buffer.append(row)
            self.samples_since_window += 1

            # Process window if enough samples
            if self.samples_since_window >= self.step and len(self.buffer) >= self.window_size:
                result = self._process_window()
                self.samples_since_window = 0
                return result

        return None

    def _process_window(self) -> Dict:
        """Process the current window: filter → features → inference."""
        t_start = time.perf_counter()

        # Get window
        buffer_arr = np.array(list(self.buffer))
        window = buffer_arr[-self.window_size:]

        # Filter
        from scipy import signal as scipy_signal
        filtered = np.zeros_like(window, dtype=np.float64)
        for ch in range(self.n_channels):
            f = scipy_signal.filtfilt(*self.filters['bandpass'], window[:, ch])
            f = scipy_signal.filtfilt(*self.filters['notch'], f)
            filtered[:, ch] = f

        # Extract features — MUST match whatever trained the loaded ONNX
        # model. Models deployed through this product are trained via
        # EMGClassifier, which uses features.py's 7-feature TIME_FEATURES
        # set (FEATURE_ORDER = MAV, RMS, ZCR, WL, SSC, WAMP, MYOP) — NOT
        # features_fast.py's ~38-feature-per-channel set. Using the wrong
        # one here previously caused a hard ONNX shape-mismatch crash on
        # every single call (confirmed: 35 features produced vs 7
        # expected) — real-time inference could not work with any model
        # trained through the normal product workflow. If you deploy a
        # model trained on a different feature set, keep this in sync.
        from ..core.features import extract_features_stream, estimate_noise_floor
        from ..core.classifier import FEATURE_ORDER

        diff_noise_floor = estimate_noise_floor(
            np.abs(np.diff(filtered[:, 0])), sampling_rate=self.fs)
        all_channel_feats = extract_features_stream(
            filtered, self.fs,
            window_size=self.window_size,
            overlap=0.0,  # we already have exactly one window
            include_freq=False,
            threshold=diff_noise_floor,
        )
        # Channel 0 only, matching EMGClassifier's default selected_channel=0
        ch0_feats = all_channel_feats[0] if all_channel_feats else []
        if ch0_feats:
            X = np.array([[f.get(name, 0.0) for name in FEATURE_ORDER]
                          for f in ch0_feats], dtype=np.float32)
        else:
            X = np.zeros((0, len(FEATURE_ORDER)), dtype=np.float32)

        # Inference
        if len(X) > 0:
            predictions = self.inference.predict(X)
            probas = self.inference.predict_proba(X)
            pred = predictions[0]
            confidence = float(np.max(probas[0])) if probas.ndim == 2 else 0.5
        else:
            pred = 'unknown'
            confidence = 0.0

        # Smoothing (majority vote)
        self.prediction_history.append((pred, confidence))
        smoothed_pred = self._smooth_predictions()

        # Performance metrics
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self.total_windows += 1
        self.total_time_ms += elapsed_ms

        return {
            'prediction': smoothed_pred,
            'raw_prediction': pred,
            'confidence': confidence,
            'processing_time_ms': elapsed_ms,
            'window_index': self.total_windows,
            'realtime': elapsed_ms < (self.step / self.fs * 1000),
        }

    def _smooth_predictions(self) -> str:
        """Majority vote over last N predictions."""
        if not self.prediction_history:
            return 'unknown'
        # Weight by confidence
        weighted = {}
        for p, c in self.prediction_history:
            weighted[p] = weighted.get(p, 0.0) + c
        return max(weighted, key=weighted.get)

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        avg_time = self.total_time_ms / max(self.total_windows, 1)
        return {
            'total_windows': self.total_windows,
            'avg_processing_time_ms': avg_time,
            'realtime_capable': avg_time < (self.step / self.fs * 1000),
            'windows_per_second': 1000 / avg_time if avg_time > 0 else 0,
        }

    def reset(self):
        """Reset buffers and counters."""
        self.buffer.clear()
        self.prediction_history.clear()
        self.samples_since_window = 0
        self.total_windows = 0
        self.total_time_ms = 0.0
