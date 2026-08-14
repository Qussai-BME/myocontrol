"""
test_shap_reports.py - Tests for src/edge/shap_reports.py.

This module had NO test coverage before this file was added. Two real
bugs were found by actually running it end-to-end (not just importing
it): (1) the SHAP output-shape handling didn't account for modern shap's
3D ndarray convention for multi-class trees, which silently sorted along
the wrong axis and produced duplicate feature names in top_features
(e.g. the same feature appearing 5 times); (2) unrelated to the module
itself, but worth locking in — confidence/trust_score must reflect the
model's actual predicted class, not whatever the caller happened to pass.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

pytest.importorskip("shap")

from src.core.classifier import EMGClassifier, generate_synthetic_emg_dataset
from src.edge.shap_reports import ShapReportGenerator


@pytest.fixture(scope="module")
def trained_classifier():
    dataset = generate_synthetic_emg_dataset(n_classes=4, n_samples_per_class=40)
    clf = EMGClassifier(model_type='xgboost')
    X = clf.prepare_features(dataset['features'])
    clf.fit(X, dataset['labels'], groups=dataset['groups'])
    return clf, X


class TestShapReportGenerator:
    def test_top_features_are_unique(self, trained_classifier):
        """Regression test for the duplicate-feature bug: with modern
        shap's 3D (samples, features, classes) ndarray output for
        multi-class XGBoost, top_features used to contain the same
        feature name repeated multiple times instead of a genuinely
        ranked, distinct list."""
        clf, X = trained_classifier
        gen = ShapReportGenerator(classifier=clf, feature_names=clf.feature_names)
        report = gen.explain_prediction(X[:1])
        names = [f['name'] for f in report.top_features]
        assert len(names) == len(set(names)), f"Duplicate feature names: {names}"
        assert set(names) == set(clf.feature_names)

    def test_confidence_matches_actual_prediction(self, trained_classifier):
        clf, X = trained_classifier
        gen = ShapReportGenerator(classifier=clf, feature_names=clf.feature_names)
        report = gen.explain_prediction(X[:1])  # auto-predict, not forced

        feat_dict = [{n: v for n, v in zip(clf.feature_names, X[0])}]
        expected_pred = clf.predict(feat_dict)[0]
        expected_conf = clf.predict_proba(feat_dict)[0][expected_pred]

        assert report.predicted_class == expected_pred
        assert abs(report.confidence - expected_conf) < 1e-6

    def test_trust_score_in_valid_range(self, trained_classifier):
        clf, X = trained_classifier
        gen = ShapReportGenerator(classifier=clf, feature_names=clf.feature_names)
        report = gen.explain_prediction(X[:5])
        for i in range(1):
            r = gen.explain_prediction(X[i:i+1])
            assert 0.0 <= r.trust_score <= 1.0

    def test_contribution_percentages_sum_near_100(self, trained_classifier):
        clf, X = trained_classifier
        gen = ShapReportGenerator(classifier=clf, feature_names=clf.feature_names)
        report = gen.explain_prediction(X[:1])
        total_pct = sum(f['contribution_pct'] for f in report.top_features)
        assert abs(total_pct - 100.0) < 1.0

    def test_to_dict_and_to_json_roundtrip(self, trained_classifier):
        clf, X = trained_classifier
        gen = ShapReportGenerator(classifier=clf, feature_names=clf.feature_names)
        report = gen.explain_prediction(X[:1])
        d = report.to_dict()
        assert d['predicted_class'] == report.predicted_class
        j = report.to_json()
        import json
        assert json.loads(j)['predicted_class'] == report.predicted_class

    def test_works_across_model_types(self):
        """The explainer-selection logic (TreeExplainer vs LinearExplainer)
        must produce valid, non-duplicated reports for every classifier
        type this product actually offers."""
        dataset = generate_synthetic_emg_dataset(n_classes=3, n_samples_per_class=30)
        for model_type in ['xgboost', 'random_forest', 'lda', 'linear_svc']:
            clf = EMGClassifier(model_type=model_type)
            X = clf.prepare_features(dataset['features'])
            clf.fit(X, dataset['labels'], groups=dataset['groups'])
            gen = ShapReportGenerator(classifier=clf, feature_names=clf.feature_names)
            report = gen.explain_prediction(X[:1])
            names = [f['name'] for f in report.top_features]
            assert len(names) == len(set(names)), f"{model_type}: duplicate names {names}"
