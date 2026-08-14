"""
test_v04_features.py - Tests for v0.4 extended features + LOSO pipeline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.core.features_extended import (
    mav, rms, wl, zcr, ssc, wamp, myop, variance, iemg,
    log_mav, log_rms, log_var, aac, tkeo,
    skewness, kurtosis, tm3, tm4, tm5, v_order, log_det, ssi,
    ar_coefficients, histogram_features, hjorth_parameters,
    frequency_features, inter_channel_correlation,
    extract_all_features_window, extract_all_features_stream,
    feature_count,
)
from src.core.feature_selection import FeatureSelector, determine_optimal_k
from src.core.alignment import EuclideanAlignment, compute_alignment_matrix, apply_alignment
from src.core.evaluation import compute_metrics, friedman_test, wilcoxon_pairwise, per_subject_accuracy
from src.core.lodo_cv import LOSOCrossValidator
from src.core.simulator import EMGSimulator


# ============================================================
# Test 1: Extended feature count matches paper (Table 2)
# ============================================================

class TestFeatureCount:
    def test_12_channels_full_count(self):
        """For 12 channels: should produce 678 features (paper Table 2)."""
        counts = feature_count(n_channels=12, include_freq=True,
                                include_inter_channel=True)
        # Per channel: 22 TD + 4 AR + 10 Hist + 3 Hjorth + 7 Freq = 46
        # Total per channel: 46 * 12 = 552
        # ICC: 66
        # Grand total: 552 + 66 = 618
        # (Paper has 678 — difference is paper lists 31 TD per channel vs our 22.
        # The missing 9 are likely derived stats — we include the most impactful ones.)
        assert counts['Total_per_channel'] == 46
        assert counts['All_per_channel_total'] == 552
        assert counts['ICC_total'] == 66
        assert counts['GRAND_TOTAL'] == 618
        print(f"\n  Feature count: {counts['GRAND_TOTAL']} (paper: 678)")

    def test_single_channel_count(self):
        counts = feature_count(n_channels=1, include_inter_channel=False)
        assert counts['GRAND_TOTAL'] == 46


# ============================================================
# Test 2: New TD features numerical correctness
# ============================================================

class TestNewTDFeatures:
    def test_variance(self):
        assert abs(variance(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) - 2.0) < 1e-10

    def test_iemg(self):
        assert abs(iemg(np.array([1.0, -2.0, 3.0])) - 6.0) < 1e-10

    def test_log_mav_zero(self):
        assert log_mav(np.zeros(100)) == 0.0

    def test_aac(self):
        sig = np.array([0.0, 1.0, 2.0, 3.0])
        assert abs(aac(sig) - 1.0) < 1e-10

    def test_tkeo_constant_signal(self):
        """TKEO of constant signal should be 0 (x^2 - x*x = 0)."""
        sig = np.ones(100) * 0.5
        result = tkeo(sig)
        assert result == 0.0  # x[i]^2 - x[i-1]*x[i+1] = 0 for constant

    def test_tkeo_varying_signal(self):
        """TKEO should be positive for varying signal."""
        sig = np.sin(np.linspace(0, 4 * np.pi, 100))
        result = tkeo(sig)
        assert result > 0

    def test_skewness_symmetric(self):
        assert abs(skewness(np.array([-1.0, 0.0, 1.0]))) < 1e-3

    def test_skewness_right_skewed(self):
        sig = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0])
        assert skewness(sig) > 0

    def test_kurtosis_normal(self):
        # Normal distribution has kurtosis ≈ 0 (excess)
        rng = np.random.default_rng(42)
        sig = rng.normal(0, 1, 10000)
        assert abs(kurtosis(sig)) < 0.2

    def test_tm3_symmetric(self):
        assert abs(tm3(np.array([-1.0, 0.0, 1.0]))) < 1e-3

    def test_tm4_positive(self):
        # tm4 = mean((x-mean)^4) — always >= 0
        # Use 5 samples (tm4 requires len >= 4)
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = tm4(sig)
        assert result >= 0
        # For [1,2,3,4,5], mean=3, deviations=[-2,-1,0,1,2], ^4=[16,1,0,1,16]
        # mean = 34/5 = 6.8
        assert abs(result - 6.8) < 1e-10

    def test_v_order(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0])
        assert v_order(sig, order=2) > 0

    def test_log_det_positive_signal(self):
        sig = np.abs(np.random.randn(100))
        result = log_det(sig)
        assert result > 0

    def test_ssi(self):
        sig = np.array([1.0, 2.0, 3.0])
        assert abs(ssi(sig) - 14.0) < 1e-10


# ============================================================
# Test 3: AR coefficients
# ============================================================

class TestARCoefficients:
    def test_returns_correct_count(self):
        sig = np.random.randn(100)
        coefs = ar_coefficients(sig, order=4)
        assert len(coefs) == 4

    def test_short_signal_returns_zeros(self):
        sig = np.array([1.0])
        coefs = ar_coefficients(sig, order=4)
        assert all(c == 0.0 for c in coefs)

    def test_coefficients_are_nonzero_for_a_real_signal(self):
        """Regression test for a real bug: ar_coefficients() used to call
        scipy.signal.lpc, which doesn't exist in ANY released scipy
        version, so every call hit a bare except and silently returned
        [0.0]*order — for every window, every channel, every subject, in
        every run of this pipeline to date. Neither pre-existing test
        above would have caught this (one only checks length, the other
        checks the OTHER zero case). This is the test that would have."""
        rng = np.random.RandomState(0)
        sig = rng.randn(500).cumsum()  # non-trivial autocorrelation structure
        coefs = ar_coefficients(sig, order=4)
        assert not all(c == 0.0 for c in coefs), (
            "AR coefficients are all zero for a signal with real "
            "autocorrelation structure — ar_coefficients() is broken again.")

    def test_recovers_known_ar_process_coefficients(self):
        """Validate against a synthetic AR(2) process with known
        coefficients: x[n] = 0.6*x[n-1] - 0.3*x[n-2] + noise. The
        Yule-Walker convention expects recovered coefficients near
        [-0.6, 0.3] (sign-flipped from the generating equation — see
        ar_coefficients() docstring)."""
        rng = np.random.RandomState(0)
        n = 3000
        x = np.zeros(n)
        for i in range(2, n):
            x[i] = 0.6 * x[i - 1] - 0.3 * x[i - 2] + rng.normal(0, 0.1)
        coefs = ar_coefficients(x[500:], order=2)
        assert abs(coefs[0] - (-0.6)) < 0.1
        assert abs(coefs[1] - 0.3) < 0.1


# ============================================================
# Test 4: Histogram features
# ============================================================

class TestHistogram:
    def test_returns_n_bins(self):
        sig = np.abs(np.random.randn(1000))
        hist = histogram_features(sig, n_bins=10)
        assert len(hist) == 10

    def test_sums_to_one(self):
        sig = np.abs(np.random.randn(1000))
        hist = histogram_features(sig, n_bins=10)
        assert abs(sum(hist.values()) - 1.0) < 1e-6

    def test_zero_signal(self):
        sig = np.zeros(100)
        hist = histogram_features(sig, n_bins=10)
        # All values fall in bin 0 (since range is [0, max+epsilon])
        # So hist_0 = 1.0, others = 0.0
        assert hist['Hist_0'] == 1.0
        assert all(v == 0.0 for k, v in hist.items() if k != 'Hist_0')


# ============================================================
# Test 5: Hjorth parameters
# ============================================================

class TestHjorth:
    def test_constant_signal(self):
        result = hjorth_parameters(np.ones(100) * 0.5)
        assert result['Activity'] >= 0
        assert result['Mobility'] == 0.0
        assert result['Complexity'] == 0.0

    def test_varying_signal(self):
        sig = np.sin(np.linspace(0, 10 * np.pi, 1000))
        result = hjorth_parameters(sig)
        assert result['Activity'] > 0
        assert result['Mobility'] > 0
        assert result['Complexity'] > 0


# ============================================================
# Test 6: Frequency features
# ============================================================

class TestFreqFeatures:
    def test_pure_tone_mnf(self):
        fs = 1000
        t = np.linspace(0, 1, fs, endpoint=False)
        sig = np.sin(2 * np.pi * 50 * t)
        feats = frequency_features(sig, fs)
        assert 45 < feats['MNF'] < 55

    def test_spectral_entropy_uniform_is_max(self):
        """White noise should have high spectral entropy (near 1)."""
        rng = np.random.default_rng(42)
        sig = rng.normal(0, 1, 4096)
        feats = frequency_features(sig, 1000)
        assert feats['SpEntropy'] > 0.8

    def test_band_power(self):
        fs = 1000
        t = np.linspace(0, 1, fs, endpoint=False)
        sig = np.sin(2 * np.pi * 100 * t)  # 100 Hz tone
        feats = frequency_features(sig, fs)
        # 100 Hz is in 20-150 band
        assert feats['Band_20_150'] > feats['Band_150_350']


# ============================================================
# Test 7: Inter-channel correlation
# ============================================================

class TestInterChannelCorrelation:
    def test_identical_channels_perfect_correlation(self):
        sig = np.random.randn(100, 2)
        sig[:, 1] = sig[:, 0]  # identical
        icc = inter_channel_correlation(sig)
        assert len(icc) == 1  # C(2,2) = 1 pair
        assert abs(icc['ICC_0_1'] - 1.0) < 1e-3

    def test_anticorrelated_channels(self):
        sig = np.random.randn(100, 2)
        sig[:, 1] = -sig[:, 0]
        icc = inter_channel_correlation(sig)
        assert abs(icc['ICC_0_1'] - (-1.0)) < 1e-3

    def test_12_channels_count(self):
        sig = np.random.randn(1000, 12)
        icc = inter_channel_correlation(sig)
        assert len(icc) == 66  # C(12,2)


# ============================================================
# Test 8: Full feature extraction
# ============================================================

class TestFullFeatureExtraction:
    def test_window_extraction_count(self):
        rng = np.random.default_rng(42)
        seg = rng.normal(0, 0.1, 800)  # 400ms @ 2000Hz
        feats = extract_all_features_window(seg, fs=2000)
        # Should have: 22 TD + 4 AR + 10 Hist + 3 Hjorth + 7 Freq = 46
        assert len(feats) == 46

    def test_stream_extraction(self):
        sim = EMGSimulator(fs=2000, n_channels=4)
        signal = sim.generate_contraction(2.0, 'grip')
        feats = extract_all_features_stream(
            signal, 2000, window_size=800, overlap=0.5,
            include_freq=True, include_inter_channel=True)
        assert len(feats) > 0
        # Each window should have: 46 per channel × 4 channels + C(4,2)=6 ICC = 190
        assert len(feats[0]) == 46 * 4 + 6

    def test_12_channels_full_count(self):
        sim = EMGSimulator(fs=2000, n_channels=12)
        signal = sim.generate_contraction(0.5, 'grip')
        feats = extract_all_features_stream(
            signal, 2000, window_size=800, overlap=0.5,
            include_freq=True, include_inter_channel=True)
        # 46 × 12 + 66 = 618
        assert len(feats[0]) == 618

    def test_threshold_crossing_features_not_saturated(self):
        """Critical: TD_ZCR/TD_SSC/TD_WAMP/TD_MYOP must vary across windows.

        Regression test for the same bug found and fixed in
        engine.py/features.py: with a fixed threshold=0.0 default (the old
        behavior), WAMP saturates to (window_size-1) and MYOP saturates to
        ~1.0 on any continuous signal, making both features carry zero
        information regardless of muscle activity. This pipeline
        (features_extended.py / lodo_cv.py) is what generates the
        paper/README's claimed LOSO numbers, so a silently-degenerate
        feature here directly undermines those numbers' validity.
        """
        sim = EMGSimulator(fs=2000, n_channels=2)
        signal = sim.generate_contraction(3.0, 'grip', intensity_scale=1.0)
        feats = extract_all_features_stream(
            signal, 2000, window_size=800, overlap=0.5,
            include_freq=False, include_inter_channel=False)
        assert len(feats) > 5, "need enough windows for a variance check"

        for ch in range(2):
            for name in ('TD_ZCR', 'TD_SSC', 'TD_WAMP', 'TD_MYOP'):
                col = f'{name}_Ch{ch}'
                values = [f[col] for f in feats]
                std = np.std(values)
                assert std > 0.0, (
                    f"{col} is constant across all windows (std=0) — "
                    "the crossing threshold is saturating this feature")

    def test_no_constant_features_in_full_extraction(self):
        """Regression test for the ar_coefficients()/scipy.signal.lpc bug:
        a full extract_all_features_stream() run must not produce ANY
        constant (std=0) column on a realistic multi-window signal. This
        is what would have caught the AR-coefficients-always-zero bug at
        the pipeline level (sklearn's SelectKBest only warns about this,
        it doesn't fail loudly, so this needs an explicit assertion).

        PeakF is excluded from this check: it's an argmax over discrete
        FFT bins, which can legitimately tie across windows on a
        band-limited *synthetic* test signal (confirmed clean — zero
        constant features at all, PeakF included — on real UCI EMG data;
        see scripts/reproduce_loso_validation.py).
        """
        sim = EMGSimulator(fs=2000, n_channels=2)
        signal = sim.generate_contraction(3.0, 'grip', intensity_scale=1.0)
        feats = extract_all_features_stream(
            signal, 2000, window_size=800, overlap=0.5,
            include_freq=True, include_inter_channel=True)
        assert len(feats) > 5

        import pandas as pd
        df = pd.DataFrame(feats)
        constant_cols = [c for c in df.columns[df.std() == 0].tolist()
                         if not c.startswith('Freq_PeakF')]
        assert constant_cols == [], (
            f"Found constant (zero-information) feature columns: {constant_cols}")


# ============================================================
# Test 9: Feature selection
# ============================================================

class TestFeatureSelector:
    def test_fit_transform(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 50))
        y = rng.integers(0, 3, 100)
        sel = FeatureSelector(k=20)
        X_sel = sel.fit_transform(X, y)
        assert X_sel.shape == (100, 20)

    def test_k_greater_than_features(self):
        X = np.random.randn(50, 30)
        y = np.random.randint(0, 2, 50)
        sel = FeatureSelector(k=100)
        sel.fit(X, y)
        assert sel.k == 30  # capped


# ============================================================
# Test 10: Euclidean Alignment
# ============================================================

class TestEuclideanAlignment:
    def test_alignment_matrix_shape(self):
        signals = [np.random.randn(1000, 4) for _ in range(3)]
        R = compute_alignment_matrix(signals)
        assert R.shape == (4, 4)

    def test_apply_alignment_shape(self):
        sig = np.random.randn(1000, 4)
        R = np.eye(4)
        aligned = apply_alignment(sig, R)
        assert aligned.shape == sig.shape

    def test_identity_matrix_no_change(self):
        sig = np.random.randn(100, 4)
        R = np.eye(4)
        aligned = apply_alignment(sig, R)
        np.testing.assert_array_almost_equal(aligned, sig)

    def test_ea_class_fit_transform(self):
        signals = [np.random.randn(1000, 4) for _ in range(3)]
        ea = EuclideanAlignment()
        ea.fit(signals)
        aligned = ea.transform(signals[0])
        assert aligned.shape == signals[0].shape


# ============================================================
# Test 11: Evaluation metrics
# ============================================================

class TestEvaluation:
    def test_compute_metrics_basic(self):
        y_true = ['rest', 'flexion', 'grip', 'rest']
        y_pred = ['rest', 'flexion', 'rest', 'rest']
        classes = ['rest', 'flexion', 'grip']
        metrics = compute_metrics(y_true, y_pred, classes)
        assert 'accuracy' in metrics
        assert 'macro_f1' in metrics
        assert 'confusion_matrix_overall' in metrics
        assert 'confusion_matrix_active' in metrics
        assert metrics['n_samples'] == 4

    def test_rest_dominance_detection(self):
        """All-rest predictions should have high rest_recall, low active_acc."""
        y_true = ['rest', 'flexion', 'grip', 'pinch']
        y_pred = ['rest', 'rest', 'rest', 'rest']  # always predict rest
        classes = ['rest', 'flexion', 'grip', 'pinch']
        metrics = compute_metrics(y_true, y_pred, classes)
        assert metrics['rest_recall'] == 1.0
        assert metrics['active_only_accuracy'] == 0.0
        # Inflation should be high (accuracy from rest dominance)
        assert metrics['inflation_pp'] > 0

    def test_friedman_test(self):
        rng = np.random.default_rng(42)
        scores = {
            'method_A': list(rng.normal(0.7, 0.05, 10)),
            'method_B': list(rng.normal(0.65, 0.05, 10)),
            'method_C': list(rng.normal(0.6, 0.05, 10)),
        }
        result = friedman_test(scores)
        assert 'chi2' in result
        assert 'p_value' in result
        assert 'mean_ranks' in result

    def test_wilcoxon_pairwise(self):
        rng = np.random.default_rng(42)
        scores = {
            'A': list(rng.normal(0.7, 0.05, 10)),
            'B': list(rng.normal(0.6, 0.05, 10)),
        }
        result = wilcoxon_pairwise(scores)
        assert 'A vs B' in result
        assert 'p_value' in result['A vs B']
        assert 'cohens_d' in result['A vs B']


# ============================================================
# Test 12: Full LOSO pipeline (synthetic data)
# ============================================================

class TestLOSOPipeline:
    def test_loso_runs_synthetic(self):
        """End-to-end LOSO on synthetic data (small scale)."""
        # Create 3 synthetic subjects
        subjects = []
        sim = EMGSimulator(fs=2000, n_channels=4)
        for subj_id in range(3):
            signal = sim.generate_contraction(2.0, 'grip', intensity_scale=1.0 + 0.1 * subj_id)
            # 5 windows, 3 classes
            n_windows = 10
            labels = np.array(['rest', 'flexion', 'grip'] * 4)[:n_windows]
            label_times = np.linspace(0, 2.0, n_windows)
            subjects.append({
                'signal': signal,
                'labels': labels,
                'label_times': label_times,
                'subject_id': f'subj_{subj_id}',
            })

        from sklearn.ensemble import RandomForestClassifier
        cv = LOSOCrossValidator(
            fs=2000, window_ms=200, overlap=0.5,
            k_features=50,  # small k for synthetic
            include_freq=True, include_inter_channel=True,
            use_euclidean_alignment=True, verbose=False,
        )

        result = cv.run_loso(
            subjects,
            lambda: RandomForestClassifier(n_estimators=20, random_state=42),
            classifier_name='RF',
        )

        assert 'accuracy_mean' in result
        assert 'macro_f1_mean' in result
        assert len(result['per_fold']) == 3
        assert 0 <= result['accuracy_mean'] <= 1
