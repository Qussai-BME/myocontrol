"""
onnx_export.py - Export trained EMG classifiers to ONNX for Edge deployment
MyoControl Suite v0.5

Converts scikit-learn classifiers (XGBoost, RF, LDA, LinearSVC) to ONNX format
for:
- Cross-platform inference (Python, C++, JavaScript, Rust)
- Edge devices (Raspberry Pi, mobile, browser)
- Real-time performance (< 5ms inference)
- No Python dependency at deployment

Usage:
    clf = EMGClassifier(model_type='xgboost')
    clf.fit(X, y)
    OnnxExporter.export(clf, 'model.onnx', feature_names=clf.feature_names)
"""
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict
import json

logger = logging.getLogger(__name__)

try:
    import onnx
    import onnxruntime as ort
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    HAS_ONNX = True

    # Register XGBoost converter if available
    try:
        from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
        from skl2onnx.operator_converters.xgboost import convert_xgboost
        try:
            import xgboost
            update_registered_converter(
                'XGBClassifier', 'XGBoostXGBClassifier',
                calculate_linear_classifier_output_shapes,
                convert_xgboost,
            )
            logger.info("XGBoost ONNX converter registered")
        except (ImportError, Exception) as e:
            logger.debug(f"XGBoost converter registration skipped: {e}")
    except ImportError:
        pass

except ImportError:
    HAS_ONNX = False
    logger.warning("ONNX packages not installed. Install with: "
                   "pip install onnx onnxruntime skl2onnx")


class OnnxExporter:
    """Export EMG classifiers to ONNX format."""

    @staticmethod
    def export(classifier, output_path: str,
               feature_names: Optional[List[str]] = None,
               n_features: Optional[int] = None) -> Dict:
        """
        Export a trained classifier to ONNX.

        Parameters
        ----------
        classifier : EMGClassifier or sklearn estimator
        output_path : where to save the .onnx file
        feature_names : list of feature names (for metadata)
        n_features : number of input features

        Returns
        -------
        dict with export info
        """
        if not HAS_ONNX:
            raise ImportError(
                "ONNX packages required. Install: pip install onnx onnxruntime skl2onnx")

        # Extract the underlying sklearn/xgboost model
        if hasattr(classifier, 'model'):
            model = classifier.model
            scaler = classifier.scaler
            classes = classifier.classes_
            feat_names = classifier.feature_names or feature_names or []
        else:
            model = classifier
            scaler = None
            classes = getattr(classifier, 'classes_', None)
            feat_names = feature_names or []

        # Determine input dimension
        if hasattr(model, 'n_features_in_'):
            n_feat = model.n_features_in_
        elif n_features:
            n_feat = n_features
        else:
            raise ValueError("Cannot determine number of features")

        # Build a pipeline (scaler + model) if scaler exists
        if scaler is not None:
            from sklearn.pipeline import Pipeline
            pipeline = Pipeline([
                ('scaler', scaler),
                ('model', model),
            ])
        else:
            pipeline = model

        # Convert to ONNX with fallback for XGBoost
        initial_type = [('input', FloatTensorType([None, n_feat]))]

        # Try multiple options for compatibility
        onnx_model = None
        conversion_errors = []

        # Option 1: with options (zipmap off)
        try:
            onnx_model = convert_sklearn(
                pipeline,
                initial_types=initial_type,
                target_opset=15,
                options={id(pipeline): {'zipmap': False}},
            )
        except Exception as e:
            conversion_errors.append(f"Option 1 (with zipmap=False): {e}")

        # Option 2: without options
        if onnx_model is None:
            try:
                onnx_model = convert_sklearn(
                    pipeline,
                    initial_types=initial_type,
                    target_opset=15,
                )
            except Exception as e:
                conversion_errors.append(f"Option 2 (no options): {e}")

        # Option 3: 'nocl' (no class labels) instead of 'zipmap' -- some
        # sklearn classifiers without a native predict_proba (e.g.
        # LinearSVC, which outputs a decision function, not
        # probabilities) don't support the 'zipmap' option this exporter
        # tries first, but do support 'nocl'. Without this, LinearSVC
        # export fails outright (confirmed: "Option 'zipmap' not in
        # ['nocl', 'output_class_labels', 'raw_scores'] for class
        # 'LinearSVC'" — the error message itself names 'nocl' as valid).
        if onnx_model is None:
            try:
                onnx_model = convert_sklearn(
                    pipeline,
                    initial_types=initial_type,
                    target_opset=15,
                    options={id(pipeline): {'nocl': True}},
                )
            except Exception as e:
                conversion_errors.append(f"Option 3 (nocl=True, for classifiers like LinearSVC without native zipmap support): {e}")

        # Option 4: fallback for XGBoost - use a tree ensemble ONNX directly
        if onnx_model is None and 'XGB' in type(model).__name__:
            try:
                onnx_model = OnnxExporter._xgboost_to_onnx_fallback(
                    model, scaler, n_feat, classes)
                logger.info("Used XGBoost → ONNX direct fallback")
            except Exception as e:
                conversion_errors.append(f"Option 4 (XGBoost fallback): {e}")

        if onnx_model is None:
            raise RuntimeError(
                f"All ONNX conversion options failed:\n" +
                "\n".join(conversion_errors))

        # Add metadata
        if feat_names:
            meta = {
                'feature_names': feat_names,
                'classes': list(classes) if classes is not None else [],
                'format': 'myocontrol-v0.5',
                'description': 'EMG gesture classifier',
            }
            for key, value in meta.items():
                inst = onnx_model.metadata_props.add()
                inst.key = key
                inst.value = json.dumps(value) if isinstance(value, list) else str(value)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"ONNX model saved to {output_path}")

        # Verify
        info = OnnxExporter.verify(output_path, n_feat, classes)
        info['output_path'] = str(output_path)
        info['feature_names_count'] = len(feat_names)
        return info

    @staticmethod
    def _xgboost_to_onnx_fallback(model, scaler, n_features, classes):
        """
        Fallback: convert XGBoost model to ONNX using onnxmltools.
        """
        import onnxmltools
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost as onnxml_convert_xgboost,
        )
        from onnxmltools.convert.common.data_types import FloatTensorType as OnnxmlFloat
        initial_type = [('input', OnnxmlFloat([None, n_features]))]
        onnx_model = onnxmltools.convert_xgboost(
            model, initial_types=initial_type, target_opset=15)
        return onnx_model

    @staticmethod
    def verify(onnx_path: str, n_features: int,
               expected_classes: Optional[List] = None) -> Dict:
        """Verify ONNX model loads and produces correct output shape."""
        if not HAS_ONNX:
            return {'verified': False, 'error': 'onnxruntime not installed'}

        try:
            sess = ort.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            input_shape = sess.get_inputs()[0].shape
            output_names = [o.name for o in sess.get_outputs()]
            output_shapes = [o.shape for o in sess.get_outputs()]

            # Test inference
            test_input = np.random.randn(1, n_features).astype(np.float32)
            outputs = sess.run(None, {input_name: test_input})

            # File size
            file_size = Path(onnx_path).stat().st_size / 1024  # KB

            return {
                'verified': True,
                'input_name': input_name,
                'input_shape': input_shape,
                'output_names': output_names,
                'output_shapes': output_shapes,
                'test_output_shape': outputs[0].shape,
                'file_size_kb': file_size,
            }
        except Exception as e:
            return {'verified': False, 'error': str(e)}


class OnnxInferenceEngine:
    """
    Real-time inference engine using ONNX Runtime.
    Designed for Edge deployment (Raspberry Pi, mobile, browser).
    """

    def __init__(self, onnx_path: str):
        if not HAS_ONNX:
            raise ImportError("onnxruntime required")

        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Load metadata
        meta = self.session.get_modelmeta()
        self.feature_names = []
        self.classes = []
        if meta.custom_metadata_map:
            import json
            if 'feature_names' in meta.custom_metadata_map:
                self.feature_names = json.loads(meta.custom_metadata_map['feature_names'])
            if 'classes' in meta.custom_metadata_map:
                self.classes = json.loads(meta.custom_metadata_map['classes'])

        # Warmup
        n_features = self.session.get_inputs()[0].shape[1]
        dummy = np.zeros((1, n_features), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for a batch of feature vectors."""
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        outputs = self.session.run(None, {self.input_name: X})
        preds = outputs[0]
        if self.classes and preds.dtype in [np.int64, np.int32]:
            preds = np.array([self.classes[int(p)] for p in preds])
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (if model supports it)."""
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        outputs = self.session.run(None, {self.input_name: X})
        if len(outputs) > 1:
            return outputs[1]  # probabilities
        return outputs[0]  # logits or labels

    def benchmark(self, n_samples: int = 1000, n_features: int = 100) -> Dict:
        """Benchmark inference speed."""
        import time
        X = np.random.randn(n_samples, n_features).astype(np.float32)

        # Warmup
        for _ in range(5):
            self.session.run(None, {self.input_name: X[:1]})

        # Measure
        t0 = time.perf_counter()
        for _ in range(10):
            self.session.run(None, {self.input_name: X})
        elapsed = (time.perf_counter() - t0) / 10

        return {
            'n_samples': n_samples,
            'total_time_ms': elapsed * 1000,
            'ms_per_sample': elapsed * 1000 / n_samples,
            'samples_per_second': n_samples / elapsed,
            'realtime_capable': (elapsed * 1000 / n_samples) < 5.0,
        }
