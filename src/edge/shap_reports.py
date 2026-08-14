"""
shap_reports.py - SHAP-based transparency reports for EMG predictions
MyoControl Suite v0.5

Generates human-readable transparency reports explaining:
- Why the model predicted a specific gesture
- Which features contributed most
- Per-channel contribution breakdown
- Confidence explanation
- Trust calibration

Required by:
- EU AI Act (2026) Article 13: transparency for high-risk AI
- FDA Software as Medical Device guidance
- Clinical deployment standards
"""
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class TransparencyReport:
    """Per-prediction transparency report."""
    predicted_class: str
    confidence: float
    top_features: List[Dict]  # [{name, shap_value, contribution_pct}]
    per_channel_contribution: Dict[str, float]
    per_feature_group_contribution: Dict[str, float]
    trust_score: float  # 0-1, calibration indicator
    explanation: str
    warnings: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class ShapReportGenerator:
    """
    Generate SHAP-based transparency reports for EMG predictions.
    """

    # Feature group prefixes
    FEATURE_GROUPS = {
        'TD': 'Time-Domain',
        'Hist': 'Histogram',
        'Hjorth': 'Hjorth Parameters',
        'Freq': 'Frequency-Domain',
        'ICC': 'Inter-Channel Correlation',
    }

    def __init__(self, classifier, feature_names: Optional[List[str]] = None):
        """
        Parameters
        ----------
        classifier : EMGClassifier (must be tree-based for SHAP TreeExplainer)
        feature_names : list of feature names
        """
        self.classifier = classifier
        if hasattr(classifier, 'feature_names'):
            self.feature_names = classifier.feature_names
        elif feature_names:
            self.feature_names = feature_names
        else:
            raise ValueError("feature_names required")

        # Setup SHAP explainer
        if not HAS_SHAP:
            raise ImportError("shap package required. Install: pip install shap")

        # Get underlying model
        model = classifier.model if hasattr(classifier, 'model') else classifier
        self.explainer = shap.TreeExplainer(model) \
            if hasattr(model, 'estimators_') or 'XGB' in type(model).__name__ \
            else shap.LinearExplainer(model, np.zeros((1, len(self.feature_names))))

    def explain_prediction(self, X: np.ndarray,
                            predicted_class: Optional[str] = None,
                            confidence: Optional[float] = None) -> TransparencyReport:
        """
        Generate transparency report for a single prediction.

        Parameters
        ----------
        X : (1, n_features) feature vector
        predicted_class : model's prediction (if None, will predict)
        confidence : model's confidence (if None, will compute)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get prediction if not provided
        if predicted_class is None:
            if hasattr(self.classifier, 'predict'):
                predicted_class = self.classifier.predict(
                    [{n: v for n, v in zip(self.feature_names, X[0])}])[0]
        if confidence is None:
            if hasattr(self.classifier, 'predict_proba'):
                probas = self.classifier.predict_proba(
                    [{n: v for n, v in zip(self.feature_names, X[0])}])[0]
                confidence = probas.get(predicted_class, 0.0)

        # Compute SHAP values
        try:
            shap_values = self.explainer.shap_values(X)
            # Regardless of which SHAP output convention this installed
            # version uses, reduce to a single 1D array of per-feature
            # SHAP values for the predicted class:
            #   - legacy (shap<0.45): list of (n_samples, n_features)
            #     arrays, one per class
            #   - modern (shap>=0.45): single ndarray, either
            #     (n_samples, n_features) for binary/regression outputs,
            #     or (n_samples, n_features, n_classes) for multi-class
            # Getting this wrong (e.g. treating a 3D ndarray as already
            # being a single sample's 1D feature vector) previously
            # caused np.argsort to sort along the wrong axis, silently
            # returning class indices instead of feature indices — which
            # showed up as the same low-index feature name (e.g. "MAV")
            # appearing multiple times in top_features.
            if isinstance(shap_values, list):
                cls_idx = 0
                if hasattr(self.classifier, 'classes_'):
                    try:
                        cls_idx = list(self.classifier.classes_).index(predicted_class)
                    except ValueError:
                        cls_idx = 0
                shap_vals = np.asarray(shap_values[cls_idx])[0]
            else:
                sv = np.asarray(shap_values)
                if sv.ndim == 3:
                    # (n_samples, n_features, n_classes)
                    cls_idx = 0
                    if hasattr(self.classifier, 'classes_'):
                        try:
                            cls_idx = list(self.classifier.classes_).index(predicted_class)
                        except ValueError:
                            cls_idx = 0
                    shap_vals = sv[0, :, cls_idx]
                elif sv.ndim == 2:
                    shap_vals = sv[0]
                else:
                    shap_vals = sv
            shap_vals = np.asarray(shap_vals, dtype=float).flatten()
            if len(shap_vals) != len(self.feature_names):
                raise ValueError(
                    f"SHAP produced {len(shap_vals)} values but there are "
                    f"{len(self.feature_names)} features — output-shape "
                    "convention not recognized for this shap/model combination.")
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            shap_vals = np.zeros(len(self.feature_names))

        # Top features
        abs_shap = np.abs(shap_vals)
        total = abs_shap.sum() + 1e-10
        contributions_pct = abs_shap / total * 100

        sorted_idx = np.argsort(-abs_shap)
        top_features = []
        for idx_int in sorted_idx[:10].tolist():  # top 10
            top_features.append({
                'name': self.feature_names[idx_int],
                'shap_value': float(shap_vals[idx_int]),
                'contribution_pct': float(contributions_pct[idx_int]),
                'direction': 'positive' if shap_vals[idx_int] > 0 else 'negative',
            })

        # Per-channel contribution
        per_channel = {}
        for i, name in enumerate(self.feature_names):
            # Parse channel from feature name (e.g., "TD_MAV_Ch3" → 3)
            if '_Ch' in name:
                ch = name.split('_Ch')[-1].split('_')[0]
                per_channel[ch] = per_channel.get(ch, 0.0) + float(abs_shap[i])
            elif name.startswith('ICC_'):
                # ICC_0_1 → contributes to channels 0 and 1
                parts = name.split('_')
                if len(parts) == 3:
                    for ch in parts[1:]:
                        per_channel[ch] = per_channel.get(ch, 0.0) + float(abs_shap[i]) / 2

        # Normalize per channel
        total_ch = sum(per_channel.values()) + 1e-10
        per_channel = {f'Ch{k}': v / total_ch * 100 for k, v in per_channel.items()}

        # Per feature group contribution
        per_group = {}
        for i, name in enumerate(self.feature_names):
            for prefix, group_name in self.FEATURE_GROUPS.items():
                if name.startswith(prefix):
                    per_group[group_name] = per_group.get(group_name, 0.0) + float(abs_shap[i])
                    break

        total_group = sum(per_group.values()) + 1e-10
        per_group = {k: v / total_group * 100 for k, v in per_group.items()}

        # Trust score (confidence calibration)
        # High confidence + high SHAP concentration = high trust
        top_3_contribution = sum(f['contribution_pct'] for f in top_features[:3])
        trust_score = float(confidence * (top_3_contribution / 100))
        trust_score = max(0.0, min(1.0, trust_score))

        # Generate explanation
        explanation = self._generate_explanation(
            predicted_class, confidence, top_features[:3], trust_score)

        # Warnings
        warnings = []
        if confidence < 0.5:
            warnings.append("Low confidence prediction — model is uncertain")
        if top_3_contribution < 30:
            warnings.append("Diffuse SHAP — no dominant features")
        if trust_score < 0.3:
            warnings.append("Low trust score — manual verification recommended")

        return TransparencyReport(
            predicted_class=predicted_class,
            confidence=float(confidence),
            top_features=top_features,
            per_channel_contribution=per_channel,
            per_feature_group_contribution=per_group,
            trust_score=trust_score,
            explanation=explanation,
            warnings=warnings,
        )

    def _generate_explanation(self, predicted_class: str,
                                confidence: float,
                                top_features: List[Dict],
                                trust_score: float) -> str:
        """Generate human-readable explanation."""
        top1 = top_features[0] if top_features else None
        top2 = top_features[1] if len(top_features) > 1 else None

        explanation = (
            f"The model predicted '{predicted_class}' with {confidence*100:.1f}% confidence. "
        )

        if top1:
            explanation += (
                f"The most influential feature was '{top1['name']}' "
                f"({top1['contribution_pct']:.1f}% of total importance, "
                f"{top1['direction']} contribution). "
            )

        if top2:
            explanation += (
                f"The second most important was '{top2['name']}' "
                f"({top2['contribution_pct']:.1f}%). "
            )

        if trust_score > 0.7:
            explanation += "This is a high-trust prediction."
        elif trust_score > 0.4:
            explanation += "This is a moderate-trust prediction."
        else:
            explanation += "This is a low-trust prediction — manual verification recommended."

        return explanation

    def batch_explain(self, X: np.ndarray, predictions: List[str],
                       confidences: List[float]) -> List[TransparencyReport]:
        """Generate reports for a batch of predictions."""
        reports = []
        for i in range(len(X)):
            report = self.explain_prediction(
                X[i:i+1], predicted_class=predictions[i],
                confidence=confidences[i])
            reports.append(report)
        return reports
