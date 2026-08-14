"""
test_features_fast.py - Tests for src/core/features_fast.py (the
Numba-accelerated "v0.5" feature pipeline). This module had NO test
coverage before this file was added, despite being marketed as the
production/edge-deployment pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.core.features_fast import (
    extract_features_fast, get_feature_names, HAS_NUMBA, benchmark,
)
from src.core.simulator import EMGSimulator


class TestFeatureNames:
    def test_count_matches_extraction_single_channel_no_freq_no_icc(self):
        names = get_feature_names(1, include_freq=False, include_inter_channel=False)
        # 18 TD + 10 Hist + 3 Hjorth = 31
        assert len(names) == 31

    def test_count_matches_extraction_with_freq_and_icc(self):
        names = get_feature_names(4, include_freq=True, include_inter_channel=True)
        # (18+10+3+4) * 4 channels + C(4,2)=6 ICC pairs = 140 + 6 = 146
        assert len(names) == 146

    def test_names_align_with_actual_output_shape(self):
        sim = EMGSimulator(fs=2000, n_channels=3)
        signal = sim.generate_contraction(2.0, 'grip')
        X, names = extract_features_fast(signal, 2000, window_size=400, overlap=0.5)
        assert X.shape[1] == len(names)


class TestExtractFeaturesFast:
    def test_output_shape(self):
        sim = EMGSimulator(fs=2000, n_channels=2)
        signal = sim.generate_contraction(2.0, 'grip')
        X, names = extract_features_fast(signal, 2000, window_size=400, overlap=0.5)
        assert X.ndim == 2
        assert X.shape[0] > 0

    def test_no_nan_or_inf(self):
        sim = EMGSimulator(fs=2000, n_channels=2)
        signal = sim.generate_contraction(2.0, 'grip')
        X, names = extract_features_fast(signal, 2000, window_size=400, overlap=0.5)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_threshold_crossing_features_not_saturated(self):
        """Regression test for a real bug found via
        scripts/reproduce_loso_validation.py: WAMP/SSC/MYOP were computed
        with a hardcoded threshold=0 in the Numba-jitted path (three
        separate occurrences of this same bug existed across
        features.py, features_extended.py, and this file — this is the
        third). A threshold of 0 saturates these features to a
        near-constant value carrying ~zero information regardless of
        muscle activity."""
        sim = EMGSimulator(fs=2000, n_channels=2)
        signal = sim.generate_contraction(3.0, 'grip', intensity_scale=1.0)
        X, names = extract_features_fast(signal, 2000, window_size=800, overlap=0.5,
                                          include_freq=False, include_inter_channel=False)
        assert X.shape[0] > 5, "need enough windows for a variance check"

        df = pd.DataFrame(X, columns=names)
        for ch in range(2):
            for name in ('TD_ZCR', 'TD_SSC', 'TD_WAMP', 'TD_MYOP'):
                col = f'{name}_Ch{ch}'
                std = df[col].std()
                assert std > 0.0, (
                    f"{col} is constant across all windows (std=0) — "
                    "the crossing threshold is saturating this feature")

    def test_no_constant_features_at_all(self):
        sim = EMGSimulator(fs=2000, n_channels=2)
        signal = sim.generate_contraction(3.0, 'grip', intensity_scale=1.0)
        X, names = extract_features_fast(signal, 2000, window_size=800, overlap=0.5,
                                          include_freq=True, include_inter_channel=True)
        df = pd.DataFrame(X, columns=names)
        # PeakF (argmax over discrete FFT bins) can legitimately tie on a
        # narrowband synthetic signal — see the equivalent test in
        # test_v04_features.py for why this is excluded, not swept under
        # the rug.
        constant_cols = [c for c in df.columns[df.std() == 0].tolist()
                         if not c.startswith('Freq_PeakF')]
        assert constant_cols == [], f"Found constant feature columns: {constant_cols}"

    def test_uses_numba_when_available(self):
        # Sanity-check the environment this test suite is actually
        # running against, so a silent fallback to the (differently
        # tested) pure-numpy path doesn't go unnoticed.
        assert HAS_NUMBA is True, (
            "numba not installed in this environment — the Numba-jitted "
            "code path in features_fast.py is NOT being exercised by "
            "this test run, only the pure-numpy fallback.")


class TestBenchmark:
    def test_benchmark_runs_and_reports_sane_numbers(self):
        result = benchmark(n_channels=4, fs=200, duration_s=2.0, window_ms=200)
        assert result['n_windows'] > 0
        assert result['n_features'] > 0
        assert result['total_time_s'] > 0
        assert result['windows_per_second'] > 0
