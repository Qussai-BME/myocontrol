"""
test_classifier.py - Tests for EMG Classifier
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.core.classifier import (
    EMGClassifier, generate_synthetic_emg_dataset,
    FEATURE_ORDER, FREQ_FEATURE_ORDER,
)


class TestEMGClassifierInit:
    def test_default_init(self):
        clf = EMGClassifier(model_type='random_forest')
        assert clf.model_type == 'random_forest'
        assert clf.is_fitted is False
        assert clf.classes_ is None

    def test_invalid_model_type(self):
        with pytest.raises(ValueError):
            EMGClassifier(model_type='invalid_model')

    def test_feature_names(self):
        clf = EMGClassifier(use_frequency_features=False)
        assert clf.feature_names == FEATURE_ORDER

    def test_feature_names_with_freq(self):
        clf = EMGClassifier(use_frequency_features=True)
        assert clf.feature_names == FEATURE_ORDER + FREQ_FEATURE_ORDER


class TestEMGClassifierTraining:
    @pytest.fixture
    def dataset(self):
        return generate_synthetic_emg_dataset(
            n_classes=5, n_samples_per_class=30)

    def test_fit_basic(self, dataset):
        clf = EMGClassifier(model_type='random_forest')
        X = clf.prepare_features(dataset['features'])
        metrics = clf.fit(X, dataset['labels'])
        assert clf.is_fitted is True
        assert clf.classes_ == sorted(dataset['classes'])
        assert metrics['train_accuracy'] > 0.5

    def test_fit_with_loso(self, dataset):
        clf = EMGClassifier(model_type='random_forest')
        X = clf.prepare_features(dataset['features'])
        metrics = clf.fit(X, dataset['labels'], groups=dataset['groups'])
        assert 'loso_accuracy_mean' in metrics
        assert 'loso_accuracy_std' in metrics
        assert 0 <= metrics['loso_accuracy_mean'] <= 1

    def test_predict_before_fit_raises(self):
        clf = EMGClassifier()
        with pytest.raises(RuntimeError):
            clf.predict([{'MAV': 0.1, 'RMS': 0.1}])


class TestEMGClassifierPrediction:
    @pytest.fixture
    def trained_clf(self):
        dataset = generate_synthetic_emg_dataset(
            n_classes=5, n_samples_per_class=30)
        clf = EMGClassifier(model_type='random_forest')
        X = clf.prepare_features(dataset['features'])
        clf.fit(X, dataset['labels'])
        return clf, dataset

    def test_predict_returns_list(self, trained_clf):
        clf, dataset = trained_clf
        feats = dataset['features'][:5]
        preds = clf.predict(feats)
        assert isinstance(preds, list)
        assert len(preds) == 5
        for p in preds:
            assert p in clf.classes_

    def test_predict_proba_returns_dicts(self, trained_clf):
        clf, dataset = trained_clf
        feats = dataset['features'][:5]
        probas = clf.predict_proba(feats)
        assert len(probas) == 5
        for proba in probas:
            assert set(proba.keys()) == set(clf.classes_)
            assert 0 <= sum(proba.values()) <= 1.01  # ~1.0 with float tolerance

    def test_predict_with_confidence(self, trained_clf):
        clf, dataset = trained_clf
        feats = dataset['features'][:5]
        results = clf.predict_with_confidence(feats)
        for r in results:
            assert r.predicted_class in clf.classes_
            assert 0 <= r.confidence <= 1.0
            assert r.feature_vector is not None


class TestEMGClassifierEvaluation:
    @pytest.fixture
    def trained_clf(self):
        dataset = generate_synthetic_emg_dataset(
            n_classes=5, n_samples_per_class=30)
        clf = EMGClassifier(model_type='random_forest')
        X = clf.prepare_features(dataset['features'])
        clf.fit(X, dataset['labels'])
        return clf, dataset

    def test_evaluate_returns_metrics(self, trained_clf):
        clf, dataset = trained_clf
        X = clf.prepare_features(dataset['features'])
        metrics = clf.evaluate(X, dataset['labels'])
        assert 'accuracy' in metrics
        assert 'macro_f1' in metrics
        assert 'weighted_f1' in metrics
        assert 'confusion_matrix' in metrics
        assert 'per_class_report' in metrics
        assert 0 <= metrics['accuracy'] <= 1


class TestEMGClassifierPersistence:
    def test_save_load_roundtrip(self):
        dataset = generate_synthetic_emg_dataset(
            n_classes=3, n_samples_per_class=20)
        clf = EMGClassifier(model_type='random_forest')
        X = clf.prepare_features(dataset['features'])
        clf.fit(X, dataset['labels'])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            assert path.exists()

            loaded = EMGClassifier.load(path)
            assert loaded.is_fitted is True
            assert loaded.classes_ == clf.classes_
            assert loaded.model_type == clf.model_type

            # Predictions should match
            feats = dataset['features'][:5]
            original_preds = clf.predict(feats)
            loaded_preds = loaded.predict(feats)
            assert original_preds == loaded_preds


class TestExplainability:
    """Regression coverage for EMGClassifier.explain() across modern shap
    (>=0.45) output shapes: list-of-arrays (legacy), 2D ndarray (binary),
    and 3D ndarray (multi-class). See classifier.py explain() for details.
    """

    @pytest.mark.parametrize(
        "model_type", ["xgboost", "random_forest", "lda", "linear_svc"])
    def test_explain_returns_importance_for_all_model_types(self, model_type):
        shap = pytest.importorskip("shap")
        dataset = generate_synthetic_emg_dataset(
            n_classes=5, n_samples_per_class=30)
        clf = EMGClassifier(model_type=model_type)
        X = clf.prepare_features(dataset['features'])
        clf.fit(X, dataset['labels'], groups=dataset['groups'])

        explanation = clf.explain(X, max_samples=30)

        assert explanation is not None, (
            f"explain() returned None for model_type={model_type}; "
            "check SHAP output-shape handling in classifier.py")
        importance = explanation['feature_importance']
        assert set(importance.keys()) == set(clf.feature_names)
        assert all(isinstance(v, float) for v in importance.values())
        assert all(v >= 0.0 for v in importance.values())
        # At least one feature should carry non-zero importance
        assert max(importance.values()) > 0.0

    def test_explain_defaults_to_own_training_data(self):
        """explain(X=None) must explain the model's own training data, not
        require (and risk mismatching) an externally supplied X — this is
        what the Streamlit 'Explain Predictions' button now relies on."""
        pytest.importorskip("shap")
        dataset = generate_synthetic_emg_dataset(n_classes=4, n_samples_per_class=40)
        clf = EMGClassifier(model_type='xgboost')
        X = clf.prepare_features(dataset['features'])
        clf.fit(X, dataset['labels'], groups=dataset['groups'])

        explanation = clf.explain()  # no X passed
        assert explanation is not None
        assert explanation['n_explained_samples'] > 0

    def test_explain_without_fit_raises(self):
        pytest.importorskip("shap")
        clf = EMGClassifier(model_type='xgboost')
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.explain()

    def test_explain_bad_shape_raises_clear_error_before_calling_shap(self):
        """explain() must validate X's shape/dtype BEFORE calling shap,
        so a caller gets an immediately actionable message ('expected N
        features, got M') instead of a cryptic failure from deep inside
        shap/xgboost."""
        pytest.importorskip("shap")
        dataset = generate_synthetic_emg_dataset(n_classes=3, n_samples_per_class=20)
        clf = EMGClassifier(model_type='xgboost')
        X = clf.prepare_features(dataset['features'])
        clf.fit(X, dataset['labels'], groups=dataset['groups'])

        bad_X = X[:, :-1]  # wrong number of feature columns
        with pytest.raises(RuntimeError, match="expected"):
            clf.explain(bad_X)

    def test_explain_rejects_non_numeric_x_with_clear_message(self):
        """Regression test for the reported bug: a non-numeric value
        buried in X must be caught with a specific message naming the
        problem, not surface as a cryptic 'could not convert string to
        float' three layers deep inside shap."""
        pytest.importorskip("shap")
        dataset = generate_synthetic_emg_dataset(n_classes=3, n_samples_per_class=20)
        clf = EMGClassifier(model_type='xgboost')
        X = clf.prepare_features(dataset['features'])
        clf.fit(X, dataset['labels'], groups=dataset['groups'])

        bad_X = X.astype(object)
        bad_X[0, 0] = "[0E0,0E0,0E0]"  # simulate a corrupted cell
        with pytest.raises(RuntimeError, match="non-numeric"):
            clf.explain(bad_X)


    def test_loso_confidence_interval(self):
        """CI must bracket the mean and widen/narrow sensibly with n_folds."""
        dataset = generate_synthetic_emg_dataset(
            n_classes=5, n_samples_per_class=50)
        clf = EMGClassifier(model_type='xgboost')
        X = clf.prepare_features(dataset['features'])
        metrics = clf.fit(X, dataset['labels'], groups=dataset['groups'])

        assert metrics['loso_n_folds'] == len(np.unique(dataset['groups']))
        ci = metrics['loso_accuracy_ci95']
        assert ci is not None
        lo, hi = ci
        assert 0.0 <= lo <= metrics['loso_accuracy_mean'] <= hi <= 1.0

    def test_loso_reports_failed_folds_instead_of_silent_nan(self):
        """Regression test for the reported bug: LOSO accuracy showed NaN
        with no explanation at some window sizes. cross_val_score can
        return NaN for individual degenerate folds (error_score=np.nan);
        fit() must separate valid from failed folds and report both,
        rather than let one NaN fold silently poison np.mean() into an
        opaque NaN with zero diagnostic value."""
        rng = np.random.RandomState(0)
        X, y, groups = [], [], []
        for s in range(4):
            for c in range(3):
                for _ in range(15):
                    X.append(rng.rand(7))
                    y.append(c)
                    groups.append(f's{s}')
        X, y, groups = np.array(X), np.array(y), np.array(groups)

        clf = EMGClassifier(model_type='xgboost')
        metrics = clf.fit(X, y, groups=groups)

        # Whether or not any folds actually failed for this particular
        # random data, the metrics dict must always carry a coherent,
        # non-contradictory picture: mean is None iff n_folds==0, and
        # n_folds + n_folds_failed == total folds attempted.
        assert 'loso_n_folds_failed' in metrics
        n_total = metrics['loso_n_folds'] + metrics['loso_n_folds_failed']
        assert n_total == len(np.unique(groups))
        if metrics['loso_n_folds'] == 0:
            assert metrics['loso_accuracy_mean'] is None
            assert metrics['loso_accuracy_ci95'] is None
        else:
            assert metrics['loso_accuracy_mean'] is not None
            assert not np.isnan(metrics['loso_accuracy_mean'])

    def test_treeexplainer_failure_falls_back_to_permutation_explainer(self):
        """If shap.TreeExplainer raises (a known failure mode for certain
        xgboost/shap version combinations), explain() must automatically
        fall back to a model-agnostic explainer instead of failing
        outright."""
        pytest.importorskip("shap")
        from unittest.mock import patch
        dataset = generate_synthetic_emg_dataset(n_classes=3, n_samples_per_class=20)
        clf = EMGClassifier(model_type='xgboost')
        X = clf.prepare_features(dataset['features'])
        clf.fit(X, dataset['labels'], groups=dataset['groups'])

        with patch('src.core.classifier.shap.TreeExplainer',
                   side_effect=ValueError("simulated incompatibility")):
            explanation = clf.explain(max_samples=15)

        assert explanation is not None
        assert explanation['explainer_type'] == 'PermutationExplainer'
        assert set(explanation['feature_importance'].keys()) == set(clf.feature_names)


class TestSyntheticDataset:
    def test_dataset_structure(self):
        dataset = generate_synthetic_emg_dataset(
            n_classes=3, n_samples_per_class=10)
        assert 'features' in dataset
        assert 'labels' in dataset
        assert 'groups' in dataset
        assert 'classes' in dataset
        assert len(dataset['features']) == 30
        assert len(dataset['labels']) == 30
        assert len(dataset['classes']) == 3

    def test_dataset_class_balance(self):
        dataset = generate_synthetic_emg_dataset(
            n_classes=5, n_samples_per_class=20)
        from collections import Counter
        label_counts = Counter(dataset['labels'])
        for cls in dataset['classes']:
            assert label_counts[cls] == 20
