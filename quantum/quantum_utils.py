"""Utility functions for Quantum Machine Learning in QuantumBind AI.

Handles feature selection, quantum angle normalization, label encoding,
and classification metrics calculation.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np

QUANTUM_FEATURE_NAMES = [
    "molecular_weight",
    "logp",
    "h_bond_donors",
    "h_bond_acceptors",
]

CLASS_NAMES = ["Weak", "Moderate", "Strong"]

LABEL_SYNONYMS: Dict[str, int] = {
    "weak": 0,
    "low": 0,
    "0": 0,
    "moderate": 1,
    "medium": 1,
    "1": 1,
    "strong": 2,
    "high": 2,
    "2": 2,
}
INT_TO_CLASS = {0: "Weak", 1: "Moderate", 2: "Strong"}


def select_quantum_features(
    records: List[Dict[str, Any]],
    feature_names: Sequence[str] = QUANTUM_FEATURE_NAMES,
) -> Tuple[np.ndarray, List[str]]:
    """Extract a small subset of 4 meaningful chemical features for quantum encoding."""
    X_list: List[List[float]] = []
    for r in records:
        row: List[float] = []
        for feat in feature_names:
            # Check both raw name and table_ prefixed name
            val = r.get(feat, r.get(f"table_{feat}", 0.0))
            row.append(float(val))
        X_list.append(row)

    return np.array(X_list, dtype=np.float64), list(feature_names)


class QuantumAngleScaler:
    """Scales feature values into the range [0, pi] for quantum angle encoding."""

    def __init__(self, target_min: float = 0.0, target_max: float = math.pi):
        self.target_min = target_min
        self.target_max = target_max
        self.feature_mins: Optional[np.ndarray] = None
        self.feature_maxs: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "QuantumAngleScaler":
        """Compute min and max from training features."""
        self.feature_mins = np.min(X, axis=0)
        self.feature_maxs = np.max(X, axis=0)
        # Prevent division by zero if a feature is constant
        ranges = self.feature_maxs - self.feature_mins
        ranges[ranges < 1e-9] = 1.0
        self.ranges = ranges
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features to [target_min, target_max]."""
        if not self.is_fitted:
            raise RuntimeError("QuantumAngleScaler must be fitted before transform.")
        normalized = (X - self.feature_mins) / self.ranges
        scaled = self.target_min + normalized * (self.target_max - self.target_min)
        return np.clip(scaled, self.target_min, self.target_max)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def encode_labels(labels: Sequence[Union[str, int]]) -> np.ndarray:
    """Map string labels ('Weak'/'Low', 'Moderate'/'Medium', 'Strong'/'High') to integers 0, 1, 2."""
    encoded = []
    for lbl in labels:
        if isinstance(lbl, (int, np.integer)):
            encoded.append(int(lbl))
        else:
            clean = str(lbl).strip().lower()
            val = LABEL_SYNONYMS.get(clean, 1)
            encoded.append(val)
    return np.array(encoded, dtype=int)


def decode_labels(int_labels: Sequence[int]) -> List[str]:
    """Map integers 0, 1, 2 back to string labels."""
    return [INT_TO_CLASS.get(int(i), "Moderate") for i in int_labels]


def calculate_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = 3,
) -> Dict[str, float]:
    """Calculate multi-class Accuracy, Precision, Recall, and F1 Score directly."""
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)
    n = len(y_t)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    accuracy = float(np.mean(y_t == y_p))

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for c in range(num_classes):
        tp = int(np.sum((y_t == c) & (y_p == c)))
        fp = int(np.sum((y_t != c) & (y_p == c)))
        fn = int(np.sum((y_t == c) & (y_p != c)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(macro_prec, 4),
        "recall": round(macro_rec, 4),
        "f1_score": round(macro_f1, 4),
    }
