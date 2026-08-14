"""
classifier.py - EMG Gesture / Movement Classification
MyoControl Suite v0.3

Provides:
- EMGClassifier: scikit-learn based classifier (XGBoost / LDA / SVC / RF)
- Trained on the 7 standard features (MAV, RMS, ZCR, WL, SSC, WAMP, MYOP)
- Supports zero-calibration mode (LOSO) and per-subject calibration
- Includes a built-in synthetic demo dataset for quick start
- Optional SHAP-based explainability

This module makes the v0.3 commercially meaningful: users can train, evaluate,
and export gesture classifiers from EMG windows.

Usage:
    clf = EMGClassifier(model_type='xgboost')
    X_train, y_train = clf.prepare_features(features_per_window, labels)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    clf.explain(X_test)  # SHAP values
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import pickle
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# --------------------------------------------------------
# sklearn imports (lazy)
# --------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, f1_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed. Classifier features disabled.")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost not installed. Falling back to RandomForest.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.info("shap not installed. Explainability disabled (optional).")


# Standard feature order (must match features.py)
FEATURE_ORDER = ['MAV', 'RMS', 'ZCR', 'WL', 'SSC', 'WAMP', 'MYOP']
FREQ_FEATURE_ORDER = ['MDF', 'MNF', 'PKF']


@dataclass
class ClassificationResult:
    """Result of a single-window classification."""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    feature_vector: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'feature_vector': self.feature_vector,
        }


class EMGClassifier:
    """
    EMG gesture / movement classifier.

    Supports multiple backends:
    - 'xgboost' (default, best for tabular features)
    - 'random_forest'
    - 'lda' (fast, interpretable, classic for EMG)
    - 'linear_svc' (fast linear baseline)
    """

    SUPPORTED_MODELS = ['xgboost', 'random_forest', 'lda', 'linear_svc']

    def __init__(self,
                 model_type: str = 'xgboost',
                 use_frequency_features: bool = False,
                 random_state: int = 42):
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for EMGClassifier. "
                "Install with: pip install scikit-learn")

        self.model_type = model_type
        self.use_frequency_features = use_frequency_features
        self.random_state = random_state
        self.feature_names = (FEATURE_ORDER +
                              (FREQ_FEATURE_ORDER if use_frequency_features else []))

        self.model = self._build_model(model_type)
        self.scaler = StandardScaler()
        self.classes_: Optional[List[str]] = None
        self.is_fitted: bool = False
        self.training_metadata: Dict = {}
        self._X_train_raw: Optional[np.ndarray] = None

    def _build_model(self, model_type: str):
        if model_type == 'xgboost':
            if not HAS_XGB:
                logger.warning("xgboost not available, falling back to RandomForest")
                return RandomForestClassifier(
                    n_estimators=200, max_depth=12,
                    random_state=self.random_state, n_jobs=-1)
            return xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, n_jobs=-1,
                eval_metric='mlogloss')
        elif model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200, max_depth=12,
                random_state=self.random_state, n_jobs=-1)
        elif model_type == 'lda':
            return LinearDiscriminantAnalysis(solver='svd')
        elif model_type == 'linear_svc':
            return LinearSVC(
                C=1.0, max_iter=5000,
                random_state=self.random_state, dual='auto')
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Supported: {self.SUPPORTED_MODELS}")

    # --------------------------------------------------------
    # Feature preparation
    # --------------------------------------------------------
    def prepare_features(self,
                         features_per_window: List[Dict[str, float]]
                         ) -> np.ndarray:
        """
        Convert list of feature dicts to a (n_windows, n_features) matrix
        in the standard feature order.
        """
        X = np.zeros((len(features_per_window), len(self.feature_names)))
        for i, feat_dict in enumerate(features_per_window):
            for j, fname in enumerate(self.feature_names):
                X[i, j] = feat_dict.get(fname, 0.0)
        return X

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    def fit(self,
            X: np.ndarray,
            y: Union[np.ndarray, List[str]],
            groups: Optional[np.ndarray] = None,
            scale: bool = True) -> Dict:
        """
        Fit classifier.

        Parameters
        ----------
        X : (n_samples, n_features) feature matrix
        y : labels
        groups : optional subject IDs for LOSO evaluation
        scale : whether to standardize features

        Returns
        -------
        dict with training metrics
        """
        y = np.array(y)
        self.classes_ = sorted(np.unique(y).tolist())

        # Keep a capped, UNSCALED sample of the training data so explain()
        # can default to explaining what the model actually learned from,
        # instead of requiring the caller to pass in a feature matrix from
        # some other, possibly mismatched, source (e.g. a different signal
        # loaded elsewhere in the app than the one used to train this model).
        _explain_cap = 500
        if len(X) > _explain_cap:
            _idx = np.random.RandomState(0).choice(len(X), _explain_cap, replace=False)
            self._X_train_raw = np.asarray(X)[_idx].copy()
        else:
            self._X_train_raw = np.asarray(X).copy()

        # Build label encoder (XGBoost requires integer labels)
        self._label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self._idx_to_label = {i: c for c, i in self._label_to_idx.items()}
        y_encoded = np.array([self._label_to_idx[c] for c in y])

        if scale:
            X = self.scaler.fit_transform(X)

        # Some models (XGBoost) require numeric labels
        if self.model_type == 'xgboost':
            self.model.fit(X, y_encoded)
        else:
            self.model.fit(X, y)
        self.is_fitted = True

        # Training accuracy (decode predictions back to original labels for XGBoost)
        train_pred = self.model.predict(X)
        if self.model_type == 'xgboost':
            train_pred = np.array([self._idx_to_label[int(p)] for p in train_pred])
        train_acc = float(accuracy_score(y, train_pred))

        metrics = {
            'train_accuracy': train_acc,
            'n_samples': len(y),
            'n_features': X.shape[1],
            'n_classes': len(self.classes_),
            'classes': self.classes_,
            'model_type': self.model_type,
        }

        # LOSO cross-validation if groups provided
        if groups is not None and len(np.unique(groups)) > 1:
            try:
                logo = LeaveOneGroupOut()
                # For XGBoost, use encoded labels (sklearn's cross_val_score
                # internally calls fit() which requires numeric labels for XGBoost)
                y_for_cv = y_encoded if self.model_type == 'xgboost' else y
                # error_score=np.nan (not the default 'raise') so ONE degenerate
                # fold — e.g. a held-out subject whose classes aren't all seen in
                # the other subjects' training data, which XGBoost in particular
                # can choke on — doesn't abort the whole evaluation. We then
                # explicitly separate valid from failed folds below, instead of
                # letting a silent NaN poison np.mean() into a NaN that looks
                # like "no result" rather than "N of M folds failed".
                cv_scores = np.asarray(cross_val_score(
                    self.model, X, y_for_cv, groups=groups,
                    cv=logo, scoring='accuracy', n_jobs=-1, error_score=np.nan))

                valid_mask = ~np.isnan(cv_scores)
                n_valid = int(valid_mask.sum())
                n_total = len(cv_scores)
                metrics['loso_per_subject'] = cv_scores.tolist()
                metrics['loso_n_folds'] = n_valid
                metrics['loso_n_folds_failed'] = n_total - n_valid

                if n_valid == 0:
                    metrics['loso_accuracy_mean'] = None
                    metrics['loso_accuracy_std'] = None
                    metrics['loso_accuracy_ci95'] = None
                    logger.warning(
                        f"All {n_total} LOSO folds failed — likely every "
                        "held-out subject has at least one class that "
                        "isn't represented in the remaining training "
                        "subjects. Try a smaller window (more windows per "
                        "class per subject) or check class coverage per "
                        "subject.")
                else:
                    valid_scores = cv_scores[valid_mask]
                    metrics['loso_accuracy_mean'] = float(np.mean(valid_scores))
                    metrics['loso_accuracy_std'] = float(np.std(valid_scores))
                    if n_total - n_valid > 0:
                        logger.warning(
                            f"{n_total - n_valid} of {n_total} LOSO folds failed "
                            "(likely a held-out subject's classes weren't fully "
                            "represented in the other subjects' training data) "
                            f"and were excluded from the {n_valid}-fold average "
                            "reported below.")

                    # 95% confidence interval via the t-distribution (not a
                    # normal-approximation z-interval): with LOSO, n_folds ==
                    # n_subjects, which is typically small (5-20), and the
                    # t-distribution correctly widens the interval to reflect
                    # that extra uncertainty rather than understating it.
                    if n_valid > 1:
                        from scipy import stats as _stats
                        sem = np.std(valid_scores, ddof=1) / np.sqrt(n_valid)
                        t_crit = _stats.t.ppf(0.975, df=n_valid - 1)
                        margin = float(t_crit * sem)
                        metrics['loso_accuracy_ci95'] = (
                            max(0.0, metrics['loso_accuracy_mean'] - margin),
                            min(1.0, metrics['loso_accuracy_mean'] + margin),
                        )
                    else:
                        metrics['loso_accuracy_ci95'] = None
            except Exception as e:
                logger.warning(f"LOSO evaluation failed: {e}")

        self.training_metadata = metrics
        return metrics

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    def predict(self,
                features_per_window: List[Dict[str, float]],
                scale: bool = True) -> List[str]:
        """Predict labels for windows."""
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call .fit() first.")
        X = self.prepare_features(features_per_window)
        if scale:
            X = self.scaler.transform(X)
        preds = self.model.predict(X)
        # Decode XGBoost predictions back to original string labels
        if self.model_type == 'xgboost':
            return [self._idx_to_label[int(p)] for p in preds]
        return [str(p) for p in preds]

    def predict_proba(self,
                      features_per_window: List[Dict[str, float]],
                      scale: bool = True) -> List[Dict[str, float]]:
        """Predict per-class probabilities for windows."""
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call .fit() first.")
        X = self.prepare_features(features_per_window)
        if scale:
            X = self.scaler.transform(X)

        # LinearSVC has no predict_proba — fallback to decision_function
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)
        else:
            decision = self.model.decision_function(X)
            if decision.ndim == 1:
                # Binary
                proba = np.column_stack([1 - decision, decision])
            else:
                # Multi-class: softmax of decision function
                proba = np.exp(decision) / np.exp(decision).sum(axis=1, keepdims=True)

        return [
            {self.classes_[i]: float(proba[j, i]) for i in range(len(self.classes_))}
            for j in range(len(proba))
        ]

    def predict_with_confidence(self,
                                features_per_window: List[Dict[str, float]]
                                ) -> List[ClassificationResult]:
        """Predict + return probabilities + confidence."""
        preds = self.predict(features_per_window)
        probas = self.predict_proba(features_per_window)

        results = []
        for i, (pred, proba) in enumerate(zip(preds, probas)):
            feat_dict = features_per_window[i]
            confidence = proba.get(pred, 0.0)
            results.append(ClassificationResult(
                predicted_class=pred,
                confidence=confidence,
                probabilities=proba,
                feature_vector=feat_dict,
            ))
        return results

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    def evaluate(self,
                 X: np.ndarray,
                 y: Union[np.ndarray, List[str]],
                 scale: bool = True) -> Dict:
        """Comprehensive evaluation on test set."""
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted.")
        y = np.array(y)
        if scale:
            X = self.scaler.transform(X)

        y_pred = self.model.predict(X)
        # Decode XGBoost predictions
        if self.model_type == 'xgboost':
            y_pred = np.array([self._idx_to_label[int(p)] for p in y_pred])

        # Metrics
        accuracy = float(accuracy_score(y, y_pred))
        macro_f1 = float(f1_score(y, y_pred, average='macro', zero_division=0))
        weighted_f1 = float(f1_score(y, y_pred, average='weighted',
                                      zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y, y_pred, labels=self.classes_)

        # Per-class report
        report = classification_report(
            y, y_pred, labels=self.classes_,
            output_dict=True, zero_division=0)

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'confusion_matrix': cm.tolist(),
            'classes': self.classes_,
            'per_class_report': report,
            'n_samples': len(y),
        }

    # --------------------------------------------------------
    # SHAP explainability (optional)
    # --------------------------------------------------------
    def explain(self,
                X: Optional[np.ndarray] = None,
                scale: bool = True,
                max_samples: int = 100) -> Optional[Dict]:
        """
        Compute SHAP values for feature importance explanation.
        Requires `shap` package.

        X : feature matrix to explain, in the SAME raw (unscaled) form
            prepare_features() returns. If None (the common case), this
            explains the model's own training data instead — global
            feature importance is a property of the fitted model, so
            explaining an unrelated feature matrix (e.g. from a different
            signal than the one used to train this classifier) would be
            misleading, not just inconvenient.
        """
        if not HAS_SHAP:
            logger.warning("shap not installed. Skipping explainability.")
            return None
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted.")

        if X is None:
            if self._X_train_raw is None:
                raise RuntimeError(
                    "No X given and no training data stored on this "
                    "classifier (it may have been loaded from an older "
                    "save() that predates this feature) — pass X explicitly.")
            X = self._X_train_raw

        # Validate BEFORE calling shap: a malformed/non-numeric X produces a
        # far more useful error here (naming the actual bad values) than
        # letting it fail deep inside shap/xgboost with a cryptic message.
        X = np.asarray(X)
        if X.dtype == object or not np.issubdtype(X.dtype, np.number):
            bad_sample = repr(X.flat[0])[:120] if X.size else "(empty)"
            raise RuntimeError(
                f"Cannot explain: X has non-numeric dtype ({X.dtype}). First "
                f"element looks like {bad_sample}. This usually means a "
                f"feature value was stored as text somewhere upstream (a "
                f"malformed CSV/.mat field) rather than a plain number — "
                f"check the data that trained this classifier.")
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise RuntimeError(
                f"Cannot explain: X has shape {X.shape}, expected "
                f"(n_samples, {len(self.feature_names)}).")
        if not np.all(np.isfinite(X)):
            n_bad = int((~np.isfinite(X)).sum())
            raise RuntimeError(
                f"Cannot explain: X contains {n_bad} NaN/Inf value(s) — "
                "clean these before training/explaining.")

        try:
            if scale:
                X = self.scaler.transform(X)

            # Subsample for speed
            if len(X) > max_samples:
                idx = np.random.choice(len(X), max_samples, replace=False)
                X = X[idx]

            explainer, shap_values = self._compute_shap_values(X)

            # Aggregate to one importance score per feature, regardless of
            # which SHAP output convention this installed version uses:
            #   - legacy (shap<0.45): list of (n_samples, n_features) arrays,
            #     one per class
            #   - modern (shap>=0.45): single ndarray, either
            #     (n_samples, n_features) for binary/regression outputs, or
            #     (n_samples, n_features, n_classes) for multi-class outputs
            if isinstance(shap_values, list):
                mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=(0, 1))
            else:
                sv = np.asarray(shap_values)
                if sv.ndim == 3:
                    # (n_samples, n_features, n_classes) -> average over
                    # samples and classes, leaving one score per feature
                    mean_abs = np.abs(sv).mean(axis=(0, 2))
                elif sv.ndim == 2:
                    mean_abs = np.abs(sv).mean(axis=0)
                else:
                    raise ValueError(
                        f"Unexpected SHAP output shape: {sv.shape}")

            if mean_abs.shape[0] != len(self.feature_names):
                raise ValueError(
                    f"SHAP returned {mean_abs.shape[0]} importance scores but "
                    f"this model has {len(self.feature_names)} features — "
                    f"the output-shape convention for this shap/model "
                    f"combination wasn't recognized correctly.")

            importance = {
                fname: float(mean_abs[i])
                for i, fname in enumerate(self.feature_names)
            }
            importance = dict(sorted(importance.items(),
                                     key=lambda x: -x[1]))

            return {
                'feature_importance': importance,
                'n_explained_samples': len(X),
                'explainer_type': type(explainer).__name__,
            }
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            raise RuntimeError(f"SHAP explanation failed ({type(e).__name__}): {e}") from e

    def _compute_shap_values(self, X: np.ndarray):
        """Try the fast, model-specific explainer first; if that fails
        (some xgboost/shap version combinations have known incompatibilities
        in the native tree-contributions path), fall back to a slower but
        model-agnostic explainer that only needs predict_proba and doesn't
        depend on that native path at all.
        """
        if self.model_type in ('xgboost', 'random_forest'):
            try:
                explainer = shap.TreeExplainer(self.model)
                return explainer, explainer.shap_values(X)
            except Exception as e:
                logger.warning(
                    f"TreeExplainer failed ({type(e).__name__}: {e}) — "
                    "falling back to a model-agnostic permutation explainer. "
                    "This is slower but doesn't depend on the tree model's "
                    "native contributions path.")
                background = shap.sample(X, min(50, len(X)))
                explainer = shap.Explainer(self.model.predict_proba, background,
                                           algorithm='permutation')
                explanation = explainer(X)
                return explainer, explanation.values
        else:
            explainer = shap.LinearExplainer(self.model, X)
            return explainer, explainer.shap_values(X)

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------
    def save(self, path: Union[str, Path]):
        """Save classifier + scaler + metadata to pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'classes_': self.classes_,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'use_frequency_features': self.use_frequency_features,
                'training_metadata': self.training_metadata,
                'label_to_idx': getattr(self, '_label_to_idx', None),
                'idx_to_label': getattr(self, '_idx_to_label', None),
                'version': '0.3.0',
            }, f)
        logger.info(f"Classifier saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EMGClassifier':
        """Load classifier from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        clf = cls(
            model_type=data['model_type'],
            use_frequency_features=data['use_frequency_features'],
        )
        clf.model = data['model']
        clf.scaler = data['scaler']
        clf.classes_ = data['classes_']
        clf.feature_names = data['feature_names']
        clf.training_metadata = data.get('training_metadata', {})
        clf._label_to_idx = data.get('label_to_idx')
        clf._idx_to_label = data.get('idx_to_label')
        clf.is_fitted = True
        return clf


# ============================================================
# Synthetic demo dataset (for testing without NinaPro)
# ============================================================

def generate_synthetic_emg_dataset(n_classes: int = 5,
                                    n_samples_per_class: int = 100,
                                    n_channels: int = 4,
                                    fs: int = 2000,
                                    window_ms: int = 200,
                                    seed: int = 42) -> Dict:
    """
    Generate synthetic EMG dataset for demo/testing.

    Each "gesture" produces distinctive feature patterns:
    - Class 0 (Rest): low MAV, low RMS
    - Class 1 (Flexion): high MAV on ch0
    - Class 2 (Extension): high MAV on ch1
    - Class 3 (Pinch): high MAV on ch2
    - Class 4 (Grip): high MAV on all channels

    Returns dict with:
    - 'features': list of feature dicts (per window)
    - 'labels': list of class names
    - 'groups': subject IDs (synthetic)
    """
    rng = np.random.default_rng(seed)
    window_size = int(window_ms * fs / 1000)

    classes = ['rest', 'flexion', 'extension', 'pinch', 'grip'][:n_classes]
    features = []
    labels = []
    groups = []

    from .features import extract_window_features

    for class_idx, class_name in enumerate(classes):
        for sample_idx in range(n_samples_per_class):
            # Generate synthetic signal per class
            signal = np.zeros((window_size, n_channels))
            base_noise = 0.02

            if class_name == 'rest':
                signal[:] = rng.normal(0, base_noise, signal.shape)
            elif class_name == 'flexion':
                signal[:, 0] = rng.normal(0, 0.15, window_size)
                signal[:, 1] = rng.normal(0, 0.05, window_size)
                signal[:, 2:] = rng.normal(0, base_noise, (window_size, n_channels - 2))
            elif class_name == 'extension':
                signal[:, 0] = rng.normal(0, 0.05, window_size)
                signal[:, 1] = rng.normal(0, 0.15, window_size)
                signal[:, 2:] = rng.normal(0, base_noise, (window_size, n_channels - 2))
            elif class_name == 'pinch':
                signal[:, 2] = rng.normal(0, 0.15, window_size)
                signal[:, :2] = rng.normal(0, 0.05, (window_size, 2))
                if n_channels > 3:
                    signal[:, 3:] = rng.normal(0, base_noise, (window_size, n_channels - 3))
            elif class_name == 'grip':
                signal[:] = rng.normal(0, 0.12, signal.shape)

            # Add envelope modulation
            envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 5 *
                                          np.linspace(0, window_ms / 1000, window_size))
            signal *= envelope[:, None]

            # Use channel 0 for features (simplified demo)
            feat = extract_window_features(signal[:, 0], fs, include_freq=False)
            features.append(feat)
            labels.append(class_name)
            groups.append(sample_idx % 10)  # 10 synthetic subjects

    return {
        'features': features,
        'labels': labels,
        'groups': np.array(groups),
        'classes': classes,
    }
