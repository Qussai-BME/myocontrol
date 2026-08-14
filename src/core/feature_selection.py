"""
feature_selection.py - SelectKBest feature selection (ANOVA F-test)
MyoControl Suite v0.4

Implements fold-level feature selection matching the paper's methodology:
- SelectKBest with ANOVA F-test
- k determined by cross-validated grid search on pilot dataset
- Per-fold fitting using ONLY training subjects' statistics (no leakage)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    ANOVA F-test based feature selection (SelectKBest).

    Used per LOSO fold — fit only on training subjects to prevent leakage.
    """

    def __init__(self, k: int = 420):
        """
        Parameters
        ----------
        k : int
            Number of top features to select (paper uses k=420 from 678 raw).
        """
        self.k = k
        self.selector: Optional[SelectKBest] = None
        self.selected_features_: Optional[List[str]] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'FeatureSelector':
        """
        Fit selector on training data.

        Parameters
        ----------
        X : (n_samples, n_features) feature matrix
        y : labels
        feature_names : list of feature names (optional, for introspection)
        """
        n_features = X.shape[1]
        if self.k > n_features:
            logger.warning(
                f"k={self.k} > n_features={n_features}. Using all features.")
            self.k = n_features

        self.selector = SelectKBest(score_func=f_classif, k=self.k)
        self.selector.fit(X, y)
        self.feature_names_ = feature_names

        # Get selected feature names if provided
        if feature_names is not None:
            mask = self.selector.get_support()
            self.selected_features_ = [feature_names[i]
                                       for i, m in enumerate(mask) if m]

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply feature selection."""
        if not self.is_fitted:
            raise RuntimeError("Selector not fitted. Call .fit() first.")
        return self.selector.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, feature_names=feature_names)
        return self.transform(X)

    def get_scores(self) -> Optional[np.ndarray]:
        """Return ANOVA F-scores of all features."""
        if not self.is_fitted:
            return None
        return self.selector.scores_

    def get_pvalues(self) -> Optional[np.ndarray]:
        """Return p-values of all features."""
        if not self.is_fitted:
            return None
        return self.selector.pvalues_


def determine_optimal_k(X_train: np.ndarray, y_train: np.ndarray,
                        k_candidates: List[int] = None,
                        cv_folds: int = 5) -> int:
    """
    Determine optimal k via cross-validated grid search.
    Used on a PILOT dataset (independent of evaluation databases) to avoid leakage.

    Default k candidates based on paper's grid search.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    if k_candidates is None:
        n_features = X_train.shape[1]
        k_candidates = [100, 200, 300, 420, 500, min(n_features, 600)]

    best_k, best_score = k_candidates[0], -1
    results = {}

    for k in k_candidates:
        if k > X_train.shape[1]:
            continue
        try:
            selector = FeatureSelector(k=k)
            X_selected = selector.fit_transform(X_train, y_train)
            clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            scores = cross_val_score(clf, X_selected, y_train, cv=cv_folds,
                                      scoring='accuracy', n_jobs=-1)
            mean_score = float(np.mean(scores))
            results[k] = mean_score
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
            logger.info(f"k={k}: cv_accuracy={mean_score:.4f}")
        except Exception as e:
            logger.warning(f"k={k} failed: {e}")

    return best_k
