"""
test_dan_baseline.py - Tests for src/core/dan_baseline.py (Lite-DAN).

This module had NO test coverage before this file was added, despite
backing specific numeric claims in README.md (parameter count, training
time, expected LOSO improvement). Tests that need PyTorch itself are
skipped (not failed) in environments without it installed — but the
"fails clearly without PyTorch" behavior is tested unconditionally,
since that's exactly the path that used to crash with a raw NameError
(see the fix in dan_baseline.py's DANClassifier.__init__).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.core import dan_baseline as dm


def _linear_params(in_f, out_f):
    return in_f * out_f + out_f


def _bn_params(n):
    return 2 * n


def _expected_dan_param_count(n_features, n_classes, n_domains, hidden_dim=64):
    """Independent re-derivation of LightweightDAN's parameter count
    directly from its architecture (feature_extractor: Linear+BN,
    Linear+BN; gesture_classifier: Linear, Linear; domain_discriminator:
    Linear, Linear) — used to cross-check DANClassifier.count_parameters()
    without needing to trust the same code twice."""
    total = 0
    total += _linear_params(n_features, 128) + _bn_params(128)
    total += _linear_params(128, hidden_dim) + _bn_params(hidden_dim)
    total += _linear_params(hidden_dim, 32) + _linear_params(32, n_classes)
    total += _linear_params(hidden_dim, 32) + _linear_params(32, n_domains)
    return total


class TestWithoutTorch:
    """These must pass regardless of whether PyTorch is installed in the
    test environment — they test the failure/guard path itself."""

    def test_module_imports_cleanly_regardless_of_torch(self):
        # Just importing this test module already imports dan_baseline
        # (see top-level import above) — if that raised, this test file
        # itself would fail to collect. Explicit assertion for clarity:
        assert hasattr(dm, 'HAS_TORCH')

    @pytest.mark.skipif(dm.HAS_TORCH, reason="only meaningful without torch installed")
    def test_clear_error_without_torch_not_a_crash(self):
        """Regression test for a real bug: DANClassifier used to crash
        with a raw `NameError: name 'torch' is not defined` inside fit()
        if PyTorch wasn't installed — despite the module's docstring
        claiming a NumPy fallback existed (it didn't). Must now raise a
        clear, actionable ImportError at construction time instead."""
        with pytest.raises(ImportError, match="PyTorch"):
            dm.DANClassifier(n_features=308, n_classes=6, n_domains=10)


class TestParameterCount:
    """These need PyTorch to actually build the model. Skipped (not
    failed) if it isn't installed in this environment."""

    def test_quick_start_example_parameter_count(self):
        """Regression test for a real inaccuracy: README.md's Quick
        Start example uses DANClassifier(n_features=308, n_classes=6,
        n_domains=10), but the "~52,649 parameters" figure quoted
        elsewhere in the README does NOT match that configuration —
        analytically, n_classes=6/n_domains=10 gives 52,880 parameters,
        not 52,649 (52,649 corresponds to a different, unspecified
        n_classes+n_domains=9 configuration). This test locks in the
        correct number for the actual documented example so the two
        don't silently drift apart again.
        """
        pytest.importorskip("torch")
        clf = dm.DANClassifier(n_features=308, n_classes=6, n_domains=10, n_epochs=1)
        # Build the underlying model without a full fit() (fast, no data needed)
        clf.model = dm.LightweightDAN(n_features=308, n_classes=6, n_domains=10,
                                       hidden_dim=clf.hidden_dim)
        actual = clf.count_parameters()
        expected = _expected_dan_param_count(308, 6, 10)
        assert actual == expected == 52880

    @pytest.mark.parametrize("n_classes,n_domains", [(2, 7), (4, 5), (6, 3)])
    def test_param_count_matches_independent_derivation(self, n_classes, n_domains):
        pytest.importorskip("torch")
        model = dm.LightweightDAN(n_features=308, n_classes=n_classes, n_domains=n_domains)
        assert model.count_parameters() == _expected_dan_param_count(308, n_classes, n_domains)


class TestFitPredict:
    def test_fit_and_predict_roundtrip(self):
        pytest.importorskip("torch")
        rng = np.random.RandomState(0)
        X = rng.rand(120, 20).astype(np.float32)
        y = rng.choice(['rest', 'fist', 'grip'], 120)
        domains = rng.choice(['s1', 's2', 's3'], 120)

        clf = dm.DANClassifier(n_features=20, n_classes=3, n_domains=3, n_epochs=3)
        clf.fit(X, y, domains=domains)
        preds = clf.predict(X[:10])
        assert len(preds) == 10
        assert set(preds).issubset({'rest', 'fist', 'grip'})

        proba = clf.predict_proba(X[:10])
        assert proba.shape == (10, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    def test_predict_before_fit_raises(self):
        pytest.importorskip("torch")
        clf = dm.DANClassifier(n_features=20, n_classes=3, n_domains=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(np.zeros((5, 20)))
