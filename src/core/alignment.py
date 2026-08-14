"""
alignment.py - Euclidean Alignment (EA) for cross-subject EMG
MyoControl Suite v0.4

Implements Euclidean Alignment per LOSO fold:
- Compute alignment matrix from TRAINING subjects' covariance only
- Apply same transformation to test subject
- Reduces inter-subject variability before feature extraction

Reference: Hahne et al. (2014) "Linear and nonlinear reductions of
electromyographic data for the control of a myoelectric prosthesis"
"""
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def compute_alignment_matrix(training_signals: list) -> np.ndarray:
    """
    Compute the Euclidean Alignment matrix from a list of training signals.

    For each training subject, compute the square root of the covariance matrix.
    Average across subjects. The alignment matrix is the inverse of this average.

    Parameters
    ----------
    training_signals : list of (N_i, C) arrays
        One per training subject.

    Returns
    -------
    R : (C, C) alignment matrix
    """
    n_channels = training_signals[0].shape[1]
    R_sum = np.zeros((n_channels, n_channels))

    for sig in training_signals:
        if sig.shape[1] != n_channels:
            continue
        # Per-subject covariance
        cov = sig.T @ sig / sig.shape[0]
        # Square root via eigendecomposition (matrix is symmetric PSD)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            sqrt_cov = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
            R_sum += sqrt_cov
        except np.linalg.LinAlgError:
            logger.warning("Eigendecomposition failed for one subject — skipping")
            continue

    R_avg = R_sum / len(training_signals)
    # Inverse of the average square-root-covariance
    try:
        R = np.linalg.inv(R_avg)
    except np.linalg.LinAlgError:
        logger.warning("Matrix inversion failed — using identity")
        R = np.eye(n_channels)

    return R


def apply_alignment(signal: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Apply alignment matrix R to a signal.
    aligned_signal = signal @ R

    Parameters
    ----------
    signal : (N, C) array
    R : (C, C) alignment matrix

    Returns
    -------
    aligned : (N, C) array
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    return signal @ R


class EuclideanAlignment:
    """
    Per-fold Euclidean Alignment for LOSO cross-validation.
    """

    def __init__(self):
        self.R: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, training_signals: list) -> 'EuclideanAlignment':
        """
        Fit alignment from training subjects only.

        Parameters
        ----------
        training_signals : list of (N_i, C) arrays
        """
        self.R = compute_alignment_matrix(training_signals)
        self.is_fitted = True
        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Apply alignment to a signal."""
        if not self.is_fitted:
            raise RuntimeError("EA not fitted. Call .fit() first.")
        return apply_alignment(signal, self.R)

    def fit_transform_fold(self, training_signals: list,
                            test_signal: np.ndarray) -> tuple:
        """
        Convenience: fit on training, transform both training list and test.
        Returns (list_of_aligned_training, aligned_test).
        """
        self.fit(training_signals)
        aligned_training = [self.transform(s) for s in training_signals]
        aligned_test = self.transform(test_signal)
        return aligned_training, aligned_test
