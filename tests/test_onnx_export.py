"""
test_onnx_export.py - Tests for src/edge/onnx_export.py.

This module had NO test coverage before this file was added, despite
backing specific numeric claims in README.md (117K samples/sec, <5ms
latency) and being one of only two supported deployment paths (the
other being Lite-DAN/PyTorch) for the "edge deployment" feature.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
skl2onnx = pytest.importorskip("skl2onnx")

from src.core.classifier import EMGClassifier, generate_synthetic_emg_dataset
from src.edge.onnx_export import OnnxExporter, OnnxInferenceEngine


@pytest.fixture(scope="module")
def dataset():
    return generate_synthetic_emg_dataset(n_classes=4, n_samples_per_class=40)


@pytest.mark.parametrize("model_type", ["xgboost", "lda", "random_forest", "linear_svc"])
class TestExportAllClassifierTypes:
    """Regression coverage for a real bug: LinearSVC export used to fail
    outright (skl2onnx's default 'zipmap' option isn't supported for
    LinearSVC, which lacks a native predict_proba) — the exporter tried
    only 'zipmap=False' and 'no options', both of which still hit the
    same underlying issue, with no fallback that actually works for this
    classifier type.
    """

    def test_export_succeeds(self, dataset, model_type, tmp_path):
        clf = EMGClassifier(model_type=model_type)
        X = clf.prepare_features(dataset['features'])
        clf.fit(X, dataset['labels'], groups=dataset['groups'])

        out_path = tmp_path / f"model_{model_type}.onnx"
        OnnxExporter.export(clf, str(out_path), feature_names=clf.feature_names)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_onnx_predictions_are_sane(self, dataset, model_type, tmp_path):
        clf = EMGClassifier(model_type=model_type)
        X = clf.prepare_features(dataset['features'])
        clf.fit(X, dataset['labels'], groups=dataset['groups'])

        out_path = tmp_path / f"model_{model_type}.onnx"
        OnnxExporter.export(clf, str(out_path), feature_names=clf.feature_names)

        engine = OnnxInferenceEngine(str(out_path))
        preds = engine.predict(X[:10])
        assert len(preds) == 10
        assert set(preds).issubset(set(clf.classes_))


class TestBenchmark:
    def test_benchmark_reports_sane_numbers(self, dataset, tmp_path):
        clf = EMGClassifier(model_type='xgboost')
        X = clf.prepare_features(dataset['features'])
        clf.fit(X, dataset['labels'], groups=dataset['groups'])

        out_path = tmp_path / "model.onnx"
        OnnxExporter.export(clf, str(out_path), feature_names=clf.feature_names)
        engine = OnnxInferenceEngine(str(out_path))

        result = engine.benchmark(n_samples=500, n_features=len(clf.feature_names))
        assert result['n_samples'] == 500
        assert result['samples_per_second'] > 0
        assert result['ms_per_sample'] > 0
        # Sanity bound, not a specific throughput claim: this varies by
        # hardware and should not be hard-coded as an exact number here
        # (see README.md's note about re-verifying throughput claims on
        # your own target hardware before quoting them).
        assert result['ms_per_sample'] < 100
