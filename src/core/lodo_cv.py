"""
lodo_cv.py - Leave-One-Subject-Out (LOSO) Cross-Validation
MyoControl Suite v0.4

Implements the strict LOSO protocol matching the paper:
- Per-fold: fit preprocessing (EA, normalization, SelectKBest) on TRAINING only
- Train classifier on aligned/selected training features
- Evaluate on aligned/selected test features
- Aggregate metrics across folds
- Compute statistical tests across folds
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
import logging
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score

from .features_extended import extract_all_features_stream
from .feature_selection import FeatureSelector
from .alignment import EuclideanAlignment
from .evaluation import compute_metrics, friedman_test, wilcoxon_pairwise

logger = logging.getLogger(__name__)


class LOSOCrossValidator:
    """
    Strict Leave-One-Subject-Out cross-validator with fold-level preprocessing.

    Per fold:
    1. Split: one subject = test, rest = train
    2. Fit Euclidean Alignment on training signals only
    3. Apply alignment to both training and test
    4. Extract features per window
    5. Fold-level Z-score normalization (fit on train only)
    6. SelectKBest feature selection (fit on train only)
    7. Train classifier
    8. Predict on test
    9. Record per-fold metrics
    """

    def __init__(self,
                 fs: int = 2000,
                 window_ms: int = 400,
                 overlap: float = 0.5,
                 k_features: int = 420,
                 include_freq: bool = True,
                 include_inter_channel: bool = True,
                 use_euclidean_alignment: bool = True,
                 verbose: bool = True):
        self.fs = fs
        self.window_size = int(window_ms * fs / 1000)
        self.overlap = overlap
        self.k_features = k_features
        self.include_freq = include_freq
        self.include_inter_channel = include_inter_channel
        self.use_euclidean_alignment = use_euclidean_alignment
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    def extract_features_for_subject(self, signal: np.ndarray,
                                       labels: Optional[np.ndarray] = None,
                                       label_times: Optional[np.ndarray] = None) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        Extract windowed features for one subject's signal.
        If labels + label_times provided, assign label per window (last label in window).
        """
        features = extract_all_features_stream(
            signal, self.fs,
            window_size=self.window_size,
            overlap=self.overlap,
            include_freq=self.include_freq,
            include_inter_channel=self.include_inter_channel,
        )

        window_labels = None
        if labels is not None and label_times is not None:
            window_step = self.window_size * (1 - self.overlap) / self.fs
            window_times = [i * window_step for i in range(len(features))]
            window_labels = []
            for wt in window_times:
                # Find last label at time <= wt
                mask = label_times <= wt
                if mask.any():
                    window_labels.append(labels[mask][-1])
                else:
                    window_labels.append(labels[0])
            window_labels = np.array(window_labels)

        return features, window_labels

    def features_to_matrix(self, features_list: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Convert list of feature dicts to (n_windows, n_features) matrix."""
        if not features_list:
            return np.array([]), []
        feature_names = sorted(features_list[0].keys())
        X = np.zeros((len(features_list), len(feature_names)))
        for i, feat_dict in enumerate(features_list):
            for j, name in enumerate(feature_names):
                X[i, j] = feat_dict.get(name, 0.0)
        return X, feature_names

    def run_loso(self,
                 subjects_data: List[Dict],
                 classifier_factory: Callable,
                 classifier_name: str = 'classifier') -> Dict:
        """
        Run full LOSO cross-validation.

        Parameters
        ----------
        subjects_data : list of dicts, each with keys:
            - 'signal': (N, C) array
            - 'labels': (L,) array of integer/string labels
            - 'label_times': (L,) array of times in seconds
            - 'subject_id': str
        classifier_factory : callable returning a fresh classifier
        classifier_name : name for reporting

        Returns
        -------
        dict with per-fold + aggregate metrics
        """
        n_subjects = len(subjects_data)
        self._log(f"Running LOSO with {n_subjects} subjects, "
                  f"window={self.window_size} samples, k={self.k_features}")

        per_fold_results = []
        per_fold_accuracy = []
        per_fold_macro_f1 = []

        # Step 1: Extract features per subject (full pipeline)
        self._log("Extracting features for all subjects...")
        all_features = []
        all_labels = []
        all_subject_ids = []

        for subj in subjects_data:
            feats, labels = self.extract_features_for_subject(
                subj['signal'], subj['labels'], subj['label_times'])
            all_features.append(feats)
            all_labels.append(labels)
            all_subject_ids.append(subj['subject_id'])

        # Step 2: LOSO loop
        for test_idx in range(n_subjects):
            test_subj_id = all_subject_ids[test_idx]
            self._log(f"\nFold {test_idx + 1}/{n_subjects}: "
                      f"test = {test_subj_id}")

            # Split
            train_indices = [i for i in range(n_subjects) if i != test_idx]

            # Apply Euclidean Alignment per fold (if enabled)
            if self.use_euclidean_alignment:
                train_signals = [subjects_data[i]['signal'] for i in train_indices]
                ea = EuclideanAlignment()
                ea.fit(train_signals)
                # Re-extract features from aligned signals (more correct than aligning features)
                train_feats_aligned = []
                train_labels_aligned = []
                for i in train_indices:
                    aligned_sig = ea.transform(subjects_data[i]['signal'])
                    feats, labels = self.extract_features_for_subject(
                        aligned_sig, subjects_data[i]['labels'],
                        subjects_data[i]['label_times'])
                    train_feats_aligned.extend(feats)
                    if labels is not None:
                        train_labels_aligned.extend(labels)

                test_aligned_sig = ea.transform(subjects_data[test_idx]['signal'])
                test_feats, test_labels = self.extract_features_for_subject(
                    test_aligned_sig, subjects_data[test_idx]['labels'],
                    subjects_data[test_idx]['label_times'])
            else:
                train_feats_aligned = []
                train_labels_aligned = []
                for i in train_indices:
                    train_feats_aligned.extend(all_features[i])
                    if all_labels[i] is not None:
                        train_labels_aligned.extend(all_labels[i])
                test_feats = all_features[test_idx]
                test_labels = all_labels[test_idx]

            # Convert to matrices
            X_train, feature_names = self.features_to_matrix(train_feats_aligned)
            y_train = np.array(train_labels_aligned)
            X_test, _ = self.features_to_matrix(test_feats)
            y_test = np.array(test_labels)

            # Replace NaN/Inf
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

            # Fold-level Z-score normalization (fit on train only)
            scaler = StandardScaler()
            X_train_norm = scaler.fit_transform(X_train)
            X_test_norm = scaler.transform(X_test)

            # SelectKBest (fit on train only)
            k = min(self.k_features, X_train.shape[1])
            selector = FeatureSelector(k=k)
            X_train_sel = selector.fit_transform(X_train_norm, y_train,
                                                  feature_names=feature_names)
            X_test_sel = selector.transform(X_test_norm)

            # Train classifier
            clf = classifier_factory()
            t0 = time.perf_counter()
            clf.fit(X_train_sel, y_train)
            train_time = time.perf_counter() - t0

            # Predict
            t0 = time.perf_counter()
            y_pred = clf.predict(X_test_sel)
            inference_time = (time.perf_counter() - t0) / len(y_pred) * 1000  # ms/sample

            # Compute metrics
            classes = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
            fold_metrics = compute_metrics(y_test, y_pred, classes)

            fold_metrics['subject_id'] = test_subj_id
            fold_metrics['train_time_s'] = train_time
            fold_metrics['inference_ms_per_sample'] = inference_time

            per_fold_results.append(fold_metrics)
            per_fold_accuracy.append(fold_metrics['accuracy'])
            per_fold_macro_f1.append(fold_metrics['macro_f1'])

            self._log(f"  Acc={fold_metrics['accuracy']*100:.2f}%, "
                      f"macro-F1={fold_metrics['macro_f1']*100:.2f}%, "
                      f"active_acc={fold_metrics['active_only_accuracy']*100:.2f}%")

        # Aggregate
        accuracy_arr = np.array(per_fold_accuracy)
        f1_arr = np.array(per_fold_macro_f1)

        # 95% CI (descriptive)
        mean_acc = float(np.mean(accuracy_arr))
        std_acc = float(np.std(accuracy_arr, ddof=1))
        ci_low = mean_acc - 1.96 * std_acc / np.sqrt(n_subjects)
        ci_high = mean_acc + 1.96 * std_acc / np.sqrt(n_subjects)

        mean_f1 = float(np.mean(f1_arr))
        std_f1 = float(np.std(f1_arr, ddof=1))

        return {
            'classifier_name': classifier_name,
            'n_subjects': n_subjects,
            'window_ms': int(self.window_size / self.fs * 1000),
            'k_features': self.k_features,
            'use_ea': self.use_euclidean_alignment,
            'per_fold': per_fold_results,
            'accuracy_mean': mean_acc,
            'accuracy_std': std_acc,
            'accuracy_ci95': [ci_low, ci_high],
            'macro_f1_mean': mean_f1,
            'macro_f1_std': std_f1,
            'per_fold_accuracy': per_fold_accuracy,
            'per_fold_macro_f1': per_fold_macro_f1,
        }


def run_multi_classifier_loso(subjects_data: List[Dict],
                               classifier_factories: Dict[str, Callable],
                               cv_config: Dict) -> Dict:
    """
    Run LOSO for multiple classifiers and compute statistical tests.

    Parameters
    ----------
    subjects_data : list of subject dicts
    classifier_factories : {name: factory_callable}
    cv_config : kwargs for LOSOCrossValidator

    Returns
    -------
    dict with per-classifier results + Friedman + Wilcoxon
    """
    cv = LOSOCrossValidator(**cv_config)
    results = {}

    for name, factory in classifier_factories.items():
        logger.info(f"\n{'='*60}\nRunning LOSO for: {name}\n{'='*60}")
        results[name] = cv.run_loso(subjects_data, factory, classifier_name=name)

    # Statistical tests across classifiers
    per_fold_scores = {name: results[name]['per_fold_accuracy']
                       for name in classifier_factories}

    friedman = friedman_test(per_fold_scores)
    wilcoxon = wilcoxon_pairwise(per_fold_scores)

    return {
        'per_classifier': results,
        'friedman_test': friedman,
        'wilcoxon_pairwise': wilcoxon,
        'cv_config': cv_config,
    }
