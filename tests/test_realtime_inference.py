"""
test_realtime_inference.py - Tests for src/edge/realtime_inference.py.

This module had NO test coverage before this file was added. That
mattered: RealtimeEMGEngine used features_fast.py's ~38-feature/channel
schema internally, while every model actually trained through this
product (EMGClassifier, used throughout the Streamlit app and
dataset_loader.py) uses features.py's 7-feature TIME_FEATURES set. The
two are incompatible — real-time inference crashed with an ONNX
shape-mismatch error on every single call against any model trained
through the normal product workflow. This would only have been caught
by actually wiring the two together end-to-end, which is what these
tests do.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
skl2onnx = pytest.importorskip("skl2onnx")
onnxruntime = pytest.importorskip("onnxruntime")

from src.core.classifier import EMGClassifier, generate_synthetic_emg_dataset
from src.edge.onnx_export import OnnxExporter
from src.edge.realtime_inference import RealtimeEMGEngine


@pytest.fixture
def exported_model_path(tmp_path):
    """A model trained the same way the rest of this product trains
    one — through EMGClassifier — then exported to ONNX. This is
    exactly the artifact a user would feed to RealtimeEMGEngine."""
    dataset = generate_synthetic_emg_dataset(n_classes=4, n_samples_per_class=40)
    clf = EMGClassifier(model_type='xgboost')
    X = clf.prepare_features(dataset['features'])
    clf.fit(X, dataset['labels'], groups=dataset['groups'])
    path = tmp_path / "model.onnx"
    OnnxExporter.export(clf, str(path), feature_names=clf.feature_names)
    return str(path), clf.classes_


class TestRealtimeEMGEngine:
    def test_processes_a_product_trained_model_without_crashing(self, exported_model_path):
        """Regression test for the feature-schema mismatch bug: this
        must not raise onnxruntime.InvalidArgument (shape mismatch)."""
        model_path, classes = exported_model_path
        engine = RealtimeEMGEngine(model_path=model_path, fs=2000, window_ms=100,
                                    overlap=0.5, n_channels=1)
        rng = np.random.RandomState(0)
        got_result = False
        for _ in range(40):
            chunk = rng.normal(0, 1e-4, (20, 1))
            result = engine.add_samples(chunk)
            if result is not None:
                got_result = True
                assert result['prediction'] in classes or result['prediction'] == 'unknown'
                assert 0.0 <= result['confidence'] <= 1.0
                assert result['processing_time_ms'] >= 0
        assert got_result, "engine never produced a prediction across 40 chunks"

    def test_multichannel_signal(self, exported_model_path):
        """Model is trained on channel-0-only features (EMGClassifier's
        default), but the engine should still accept a multi-channel
        input buffer without crashing — it must consistently select the
        same channel it filters/extracts from."""
        model_path, classes = exported_model_path
        engine = RealtimeEMGEngine(model_path=model_path, fs=2000, window_ms=100,
                                    overlap=0.5, n_channels=4)
        rng = np.random.RandomState(1)
        for _ in range(40):
            chunk = rng.normal(0, 1e-4, (20, 4))
            engine.add_samples(chunk)  # must not raise
        stats = engine.get_performance_stats()
        assert stats['total_windows'] > 0

    def test_performance_stats_shape(self, exported_model_path):
        model_path, _ = exported_model_path
        engine = RealtimeEMGEngine(model_path=model_path, fs=2000, window_ms=100,
                                    overlap=0.5, n_channels=1)
        rng = np.random.RandomState(2)
        for _ in range(30):
            engine.add_samples(rng.normal(0, 1e-4, (20, 1)))
        stats = engine.get_performance_stats()
        assert set(stats.keys()) >= {'total_windows', 'avg_processing_time_ms',
                                      'realtime_capable', 'windows_per_second'}
        assert stats['avg_processing_time_ms'] >= 0

    def test_reset_clears_state(self, exported_model_path):
        model_path, _ = exported_model_path
        engine = RealtimeEMGEngine(model_path=model_path, fs=2000, window_ms=100,
                                    overlap=0.5, n_channels=1)
        rng = np.random.RandomState(3)
        for _ in range(30):
            engine.add_samples(rng.normal(0, 1e-4, (20, 1)))
        assert engine.total_windows > 0
        engine.reset()
        assert engine.total_windows == 0
        assert len(engine.buffer) == 0
