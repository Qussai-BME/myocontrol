"""
evaluation.py - Comprehensive evaluation metrics for EMG classification
MyoControl Suite v0.4

Implements the full evaluation suite matching the paper:
- Accuracy + macro-F1 + weighted-F1
- Confusion matrix (overall + active-only)
- Rest-class dominance analysis
- Per-class F1 scores
- Per-subject accuracy distribution
- Friedman test + Nemenyi post-hoc
- Wilcoxon pairwise + Cohen's d
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_recall_fscore_support,
)
from scipy import stats as scipy_stats
import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    classes: List[str],
                    rest_class: str = 'rest') -> Dict:
    """
    Compute comprehensive classification metrics.

    Includes:
    - Overall accuracy, macro-F1, weighted-F1
    - Rest recall and active-only accuracy
    - Accuracy inflation (overall - active-only)
    - Per-class F1
    - Confusion matrices (overall + active-only)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Standard metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro',
                               zero_division=0, labels=classes))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted',
                                  zero_division=0, labels=classes))

    # Per-class F1
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    per_class = {classes[i]: {
        'precision': float(p[i]),
        'recall': float(r[i]),
        'f1': float(f1[i]),
        'support': int(support[i]),
    } for i in range(len(classes))}

    # Rest class analysis
    rest_recall = 0.0
    if rest_class in classes:
        rest_idx = classes.index(rest_class)
        rest_recall = float(r[rest_idx])

    # Active-only (exclude rest class)
    active_mask = y_true != rest_class
    if active_mask.sum() > 0:
        active_acc = float(accuracy_score(
            y_true[active_mask], y_pred[active_mask]))
        active_classes = [c for c in classes if c != rest_class]
        active_f1 = float(f1_score(
            y_true[active_mask], y_pred[active_mask],
            average='macro', zero_division=0, labels=active_classes)) \
            if active_classes else 0.0
    else:
        active_acc = 0.0
        active_f1 = 0.0

    # Inflation = how much overall accuracy exceeds active accuracy
    inflation = accuracy - active_acc

    # Confusion matrices
    cm_overall = confusion_matrix(y_true, y_pred, labels=classes)

    active_indices = [i for i, c in enumerate(classes) if c != rest_class]
    if active_indices and rest_class in classes:
        cm_active = confusion_matrix(
            y_true[active_mask], y_pred[active_mask],
            labels=[c for c in classes if c != rest_class])
    else:
        cm_active = cm_overall

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'rest_recall': rest_recall,
        'active_only_accuracy': active_acc,
        'active_only_macro_f1': active_f1,
        'inflation_pp': inflation,
        'per_class': per_class,
        'confusion_matrix_overall': cm_overall.tolist(),
        'confusion_matrix_active': cm_active.tolist(),
        'n_samples': len(y_true),
        'n_classes': len(classes),
    }


# ============================================================
# Statistical tests (Friedman + Nemenyi + Wilcoxon)
# ============================================================

def friedman_test(per_fold_scores: Dict[str, List[float]]) -> Dict:
    """
    Friedman test with Iman-Davenport F-statistic.

    Parameters
    ----------
    per_fold_scores : {method_name: [score_fold1, score_fold2, ...]}

    Returns
    -------
    dict with chi2, p_value, F_statistic, F_p_value, decision
    """
    methods = list(per_fold_scores.keys())
    n_folds = len(next(iter(per_fold_scores.values())))

    # Build (n_folds, n_methods) matrix
    matrix = np.array([per_fold_scores[m] for m in methods]).T  # (folds, methods)

    # Friedman chi-square
    try:
        chi2, p_value = scipy_stats.friedmanchisquare(*[per_fold_scores[m] for m in methods])
    except Exception as e:
        logger.error(f"Friedman test failed: {e}")
        return {'error': str(e)}

    # Iman-Davenport F-statistic
    n_methods = len(methods)
    df1 = n_methods - 1
    df2 = (n_folds - 1) * (n_methods - 1)
    F_stat = ((n_folds - 1) * chi2) / (n_folds * (n_methods - 1) - chi2) \
             if (n_folds * (n_methods - 1) - chi2) > 0 else float('inf')
    F_p = float(scipy_stats.f.sf(F_stat, df1, df2))

    # Compute ranks (1=best)
    ranks = np.zeros_like(matrix)
    for i in range(n_folds):
        order = np.argsort(-matrix[i])  # descending (best first)
        for rank, method_idx in enumerate(order):
            ranks[i, method_idx] = rank + 1
    mean_ranks = {methods[i]: float(np.mean(ranks[:, i]))
                  for i in range(n_methods)}

    # Nemenyi critical difference
    # CD = q_alpha * sqrt(k(k+1)/(6N))
    # q_alpha for alpha=0.05, k methods — table lookup
    q_alpha_table = {2: 2.343, 3: 2.652, 4: 2.910, 5: 3.102,
                     6: 3.261, 7: 3.396, 8: 3.516}
    q_alpha = q_alpha_table.get(n_methods, 3.0)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_folds))

    return {
        'chi2': float(chi2),
        'p_value': float(p_value),
        'F_statistic': float(F_stat),
        'F_p_value': F_p,
        'decision': '***significant' if p_value < 0.05 else 'n.s.',
        'n_folds': n_folds,
        'n_methods': n_methods,
        'mean_ranks': mean_ranks,
        'nemenyi_CD': float(cd),
    }


def wilcoxon_pairwise(per_fold_scores: Dict[str, List[float]],
                      alpha: float = 0.05,
                      correction: str = 'holm-sidak') -> Dict:
    """
    Pairwise Wilcoxon signed-rank tests with multiple-comparison correction.

    Returns dict of {pair: {p_value, adjusted_p, significant, cohens_d, interpretation}}.
    """
    from itertools import combinations
    methods = list(per_fold_scores.keys())
    pairs = list(combinations(methods, 2))

    results = {}
    raw_p_values = []
    pair_names = []

    for m1, m2 in pairs:
        s1 = np.array(per_fold_scores[m1])
        s2 = np.array(per_fold_scores[m2])
        try:
            stat, p = scipy_stats.wilcoxon(s1, s2)
        except Exception:
            p = 1.0

        # Cohen's d effect size (paired)
        diff = s1 - s2
        d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))
        if abs(d) < 0.2:
            interp = 'negligible'
        elif abs(d) < 0.5:
            interp = 'small'
        elif abs(d) < 0.8:
            interp = 'medium'
        else:
            interp = 'large'

        pair_name = f'{m1} vs {m2}'
        pair_names.append(pair_name)
        raw_p_values.append(float(p))
        results[pair_name] = {
            'p_value': float(p),
            'cohens_d': d,
            'interpretation': interp,
        }

    # Apply multiple-comparison correction
    n_tests = len(raw_p_values)
    sorted_indices = np.argsort(raw_p_values)
    for rank, idx in enumerate(sorted_indices):
        if correction == 'holm-sidak':
            adjusted_p = min(1.0, raw_p_values[idx] * (n_tests - rank))
        elif correction == 'bonferroni':
            adjusted_p = min(1.0, raw_p_values[idx] * n_tests)
        else:  # no correction
            adjusted_p = raw_p_values[idx]
        results[pair_names[idx]]['adjusted_p'] = float(adjusted_p)
        results[pair_names[idx]]['significant'] = bool(adjusted_p < alpha)

    return results


# ============================================================
# Per-subject variability
# ============================================================

def per_subject_accuracy(per_subject_results: Dict[str, List[float]]) -> Dict:
    """
    Analyze per-subject accuracy distribution.

    Parameters
    ----------
    per_subject_results : {method: [acc_subj1, acc_subj2, ...]}
    """
    summary = {}
    for method, accs in per_subject_results.items():
        accs = np.array(accs)
        summary[method] = {
            'min': float(np.min(accs)),
            'max': float(np.max(accs)),
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
            'range_pp': float(np.max(accs) - np.min(accs)),
            'min_subject_idx': int(np.argmin(accs)),
            'max_subject_idx': int(np.argmax(accs)),
        }
    return summary
