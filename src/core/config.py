"""
config.py - EMG Configuration Dataclass
MyoControl Suite v0.3
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class EMGConfig:
    """Configuration parameters following biomedical standards (IEEE/ISEK)."""
    # Sampling
    sampling_rate: int = 2000

    # Bandpass filter
    cutoff_low: float = 20.0       # Hz (high-pass)
    cutoff_high: float = 450.0     # Hz (low-pass)
    filter_order: int = 4
    filter_type: str = 'butterworth'  # butterworth, chebyshev, bessel, elliptic

    # Notch filter (powerline interference)
    notch_freq: float = 50.0
    notch_quality: float = 30.0

    # Windowing
    window_size: int = 200         # samples (100ms @ 2000Hz)
    overlap: float = 0.5

    # Noise estimation
    noise_estimation_method: str = 'percentile'  # percentile, median, manual
    noise_percentile: float = 5.0
    manual_noise_floor: Optional[float] = None

    # Spectral analysis
    psd_method: str = 'welch'      # welch, fft
    psd_nperseg: int = 256

    # Chunking (memory management)
    chunk_duration: Optional[float] = None  # seconds, None = no chunking

    # Artifact detection
    artifact_threshold: float = 5.0  # × std above baseline

    # Threshold-crossing feature sensitivity (WAMP, MYOP, ZCR, SSC).
    # These features are only informative if the crossing threshold is set
    # relative to the noise floor — a threshold of 0 makes them saturate
    # (WAMP/SSC compare against the sample-to-sample derivative, so they
    # need a threshold on that same scale; MYOP/ZCR compare against raw
    # amplitude). threshold = feature_threshold_multiplier x the noise
    # floor of the *derivative* signal (estimated with the same
    # percentile method used for SNR), which empirically keeps all four
    # features non-degenerate across gestures and intensities. Rationale
    # follows the noise-relative thresholding recommended in
    # Zardoshti-Kermani et al. (1995) and Phinyomark et al. (2012), "EMG
    # feature evaluation for improving myoelectric pattern recognition
    # robustness".
    feature_threshold_multiplier: float = 1.0

    # SNR quality bands (dB) used only to LABEL the signal as
    # poor/fair/good/excellent for the clinician-facing summary. These are
    # a common informal rule-of-thumb for biosignal SNR (comparable in
    # spirit to the quality bands used for audio/telemetry SNR), NOT
    # values calibrated against expert-rated EMG recordings. If you have a
    # labeled dataset with expert quality ratings, refit these three
    # numbers against it before relying on the label clinically — as
    # shipped, treat the underlying SNR (dB) as the trustworthy number and
    # the poor/fair/good/excellent label as a rough, uncalibrated guide.
    snr_threshold_poor: float = 5.0
    snr_threshold_fair: float = 15.0
    snr_threshold_good: float = 25.0

    # Feature extraction options
    compute_frequency_features: bool = False

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if self.cutoff_high >= self.sampling_rate / 2:
            raise ValueError(
                f"Cutoff high ({self.cutoff_high}) must be < Nyquist "
                f"({self.sampling_rate / 2})"
            )
        if self.cutoff_low >= self.cutoff_high:
            raise ValueError(
                f"Cutoff low ({self.cutoff_low}) must be < cutoff high ({self.cutoff_high})"
            )
        if self.filter_order < 1 or self.filter_order > 10:
            raise ValueError("Filter order must be 1-10")
        if not (0 < self.overlap < 1):
            raise ValueError("Overlap must be between 0 and 1")
        if self.window_size < 10:
            raise ValueError("Window size must be >= 10 samples")
        if self.filter_type not in ['butterworth', 'chebyshev', 'bessel', 'elliptic']:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EMGConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
