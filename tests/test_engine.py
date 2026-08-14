"""
test_engine.py - Tests for EMG Engine
Run: pytest tests/test_engine.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.core.config import EMGConfig
from src.core.engine import EMGEngine
from src.core.simulator import EMGSimulator
from src.core.features import (
    mav, rms, zcr, wl, ssc, wamp, myop,
    mdf, mnf, fatigue_index,
)


# ============================================================
# Feature tests
# ============================================================

class TestTimeDomainFeatures:
    def test_mav_zero_signal(self):
        assert mav(np.zeros(100)) == 0.0

    def test_mav_constant_signal(self):
        assert mav(np.ones(100)) == 1.0

    def test_rms_zero_signal(self):
        assert rms(np.zeros(100)) == 0.0

    def test_rms_unity_signal(self):
        assert rms(np.ones(100)) == 1.0

    def test_zcr_no_crossings(self):
        assert zcr(np.ones(100)) == 0.0

    def test_zcr_alternating(self):
        sig = np.array([1, -1, 1, -1, 1])
        assert zcr(sig) > 0.5  # Most samples cross zero

    def test_wl_zero_signal(self):
        assert wl(np.zeros(100)) == 0.0

    def test_wl_constant_signal(self):
        assert wl(np.ones(100)) == 0.0

    def test_wl_ramp(self):
        sig = np.arange(100, dtype=float)
        assert wl(sig) > 0

    def test_ssc_no_changes(self):
        assert ssc(np.ones(100)) == 0.0

    def test_ssc_alternating(self):
        sig = np.array([1, -1, 1, -1, 1], dtype=float)
        assert ssc(sig) > 0.3

    def test_wamp_zero(self):
        assert wamp(np.zeros(100), threshold=0.1) == 0.0

    def test_myop_zero(self):
        assert myop(np.zeros(100), threshold=0.1) == 0.0

    def test_myop_all_above_threshold(self):
        assert myop(np.ones(100), threshold=0.5) == 1.0


class TestFrequencyDomainFeatures:
    def test_mdf_pure_tone(self):
        fs = 1000
        t = np.linspace(0, 1, fs)
        sig = np.sin(2 * np.pi * 50 * t)  # 50 Hz pure tone
        result = mdf(sig, fs, nperseg=256)
        assert 40 < result < 60  # Should be near 50 Hz

    def test_mnf_pure_tone(self):
        fs = 1000
        t = np.linspace(0, 1, fs)
        sig = np.sin(2 * np.pi * 50 * t)
        result = mnf(sig, fs, nperseg=256)
        assert 40 < result < 60

    def test_mdf_zero_signal(self):
        assert mdf(np.zeros(100), 1000) == 0.0

    def test_mnf_zero_signal(self):
        assert mnf(np.zeros(100), 1000) == 0.0


class TestFatigueIndex:
    def test_fatigue_index_increasing_signal(self):
        rms_vals = np.linspace(0.1, 0.5, 20)
        fs_features = 10  # 10 windows/sec
        fi = fatigue_index(rms_vals, fs_features)
        assert fi < 0  # Negative because RMS is increasing

    def test_fatigue_index_decreasing_signal(self):
        rms_vals = np.linspace(0.5, 0.1, 20)
        fi = fatigue_index(rms_vals, 10)
        assert fi > 0  # Positive = fatiguing

    def test_fatigue_index_short_signal(self):
        assert fatigue_index(np.array([0.1]), 10) == 0.0


# ============================================================
# Engine tests
# ============================================================

class TestEMGEngine:
    @pytest.fixture
    def engine(self):
        config = EMGConfig(sampling_rate=2000)
        return EMGEngine(config)

    @pytest.fixture
    def sample_signal(self):
        sim = EMGSimulator(fs=2000, n_channels=4)
        return sim.generate_contraction(2.0, 'flexion')

    def test_preprocess_returns_2d(self, engine, sample_signal):
        filtered = engine.preprocess(sample_signal)
        assert filtered.ndim == 2
        assert filtered.shape == sample_signal.shape

    def test_preprocess_1d_input(self, engine):
        sig_1d = np.random.randn(2000)
        filtered = engine.preprocess(sig_1d)
        assert filtered.ndim == 2
        assert filtered.shape[1] == 1

    def test_process_returns_expected_keys(self, engine, sample_signal):
        result = engine.process(sample_signal)
        for key in ['metadata', 'signal_quality', 'clinical_interpretation',
                    'time_series', 'summary_statistics']:
            assert key in result

    def test_process_fatigue_index_is_real(self, engine, sample_signal):
        """Critical: fatigue_index must NOT be hardcoded 0.0"""
        result = engine.process(sample_signal)
        ch_summary = result['summary_statistics']['channel_0']
        # On a sustained contraction, fatigue_index should be non-zero
        # (could be positive or negative depending on signal)
        assert 'fatigue_index' in ch_summary
        assert isinstance(ch_summary['fatigue_index'], float)

    def test_process_artifact_detection_runs(self, engine, sample_signal):
        """Critical: artifact_detected must be actually computed, not always False"""
        result = engine.process(sample_signal)
        assert 'artifact_detected' in result['signal_quality']
        assert 'artifact_percentage' in result['signal_quality']
        assert 'artifact_samples' in result['signal_quality']

    def test_process_with_benchmark(self, engine, sample_signal):
        result = engine.process(sample_signal, measure_time=True)
        assert 'benchmark' in result
        assert 'processing_time_ms' in result['benchmark']
        assert result['benchmark']['processing_time_ms'] > 0

    def test_process_threshold_features_not_saturated(self, engine, sample_signal):
        """Critical: WAMP/MYOP/ZCR/SSC must vary across windows.

        With threshold=0 (the old default), WAMP saturates to
        window_size-1 and MYOP saturates to ~1.0 on any continuous
        signal, making both features useless for classification (their
        std is 0). engine.process() must supply a noise-relative
        threshold (EMGConfig.feature_threshold_multiplier) so all four
        threshold-crossing features carry real information.
        """
        result = engine.process(sample_signal)
        feats = result['time_series']['features'][0]
        assert len(feats) > 5, "need enough windows for a variance check"
        df_std = {
            name: np.std([f[name] for f in feats])
            for name in ('ZCR', 'SSC', 'WAMP', 'MYOP')
        }
        for name, std in df_std.items():
            assert std > 0.0, (
                f"{name} is constant across all windows (std=0) — "
                "the crossing threshold is saturating this feature")

    def test_process_multichannel(self, engine):
        sim = EMGSimulator(fs=2000, n_channels=8)
        sig = sim.generate_contraction(1.0, 'grip')
        result = engine.process(sig)
        assert result['metadata']['n_channels'] == 8

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            EMGConfig(sampling_rate=-1).validate()

    def test_invalid_filter_raises(self):
        with pytest.raises(ValueError):
            EMGConfig(filter_type='invalid').validate()


# ============================================================
# Simulator tests
# ============================================================

class TestEMGSimulator:
    def test_generate_contraction_shape(self):
        sim = EMGSimulator(fs=2000, n_channels=4)
        sig = sim.generate_contraction(1.0, 'flexion')
        assert sig.shape == (2000, 4)

    def test_generate_contraction_multichannel(self):
        sim = EMGSimulator(fs=2000, n_channels=8)
        sig = sim.generate_contraction(0.5, 'grip')
        assert sig.shape == (1000, 8)

    def test_unknown_gesture_raises(self):
        sim = EMGSimulator()
        with pytest.raises(ValueError):
            sim.generate_contraction(1.0, 'unknown_gesture')

    def test_rest_has_low_energy(self):
        sim = EMGSimulator(fs=2000, n_channels=4)
        rest = sim.generate_contraction(1.0, 'rest')
        grip = sim.generate_contraction(1.0, 'grip')
        assert np.std(rest) < np.std(grip)

    def test_generate_sequence(self):
        sim = EMGSimulator(fs=2000, n_channels=4)
        sig = sim.generate_sequence([
            {'gesture': 'flexion', 'duration': 0.5},
            {'gesture': 'grip', 'duration': 0.5},
        ])
        assert sig.shape[0] > 2000  # at least 1 second of data
