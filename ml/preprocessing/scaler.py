"""Feature scaling module for QuantumBind AI."""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    from sklearn.preprocessing import StandardScaler
    _ = StandardScaler().fit([[1.0, 2.0], [3.0, 4.0]])
    HAS_SKLEARN = True
except (ImportError, Exception):
    StandardScaler = None
    HAS_SKLEARN = False


class FeatureScaler:
    """Standardizes numerical feature vectors (mean=0, unit variance).

    Supports scikit-learn's StandardScaler when available, with a deterministic
    self-contained fallback calculation to prevent data leakage.
    """

    def __init__(self):
        self.use_sklearn = HAS_SKLEARN
        self._sklearn_scaler = StandardScaler() if HAS_SKLEARN else None
        self.means: List[float] = []
        self.stds: List[float] = []
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: List[List[float]], feature_names: Optional[List[str]] = None) -> "FeatureScaler":
        """Compute mean and standard deviation from training matrix X."""
        if not X or not X[0]:
            raise ValueError("Cannot fit scaler on empty feature matrix.")

        num_samples = len(X)
        num_features = len(X[0])
        self.feature_names = feature_names or [f"feat_{i}" for i in range(num_features)]

        if self.use_sklearn:
            if HAS_NUMPY:
                X_arr = np.array(X, dtype=float)
            else:
                X_arr = X
            self._sklearn_scaler.fit(X_arr)
            self.means = [float(m) for m in self._sklearn_scaler.mean_]
            self.stds = [float(s) for s in self._sklearn_scaler.scale_]
            self.is_fitted = True
            return self

        # Deterministic fallback calculation
        self.means = []
        self.stds = []
        for col_idx in range(num_features):
            col_vals = [X[row_idx][col_idx] for row_idx in range(num_samples)]
            col_mean = sum(col_vals) / num_samples
            variance = sum((v - col_mean) ** 2 for v in col_vals) / num_samples
            col_std = math.sqrt(variance)
            # Avoid division by zero for constant features
            if col_std < 1e-9:
                col_std = 1.0
            self.means.append(col_mean)
            self.stds.append(col_std)

        self.is_fitted = True
        return self

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Scale matrix X using previously fitted training statistics."""
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before calling transform.")

        if self.use_sklearn:
            if HAS_NUMPY:
                X_arr = np.array(X, dtype=float)
            else:
                X_arr = X
            scaled = self._sklearn_scaler.transform(X_arr)
            return [[float(v) for v in row] for row in scaled]

        scaled_matrix: List[List[float]] = []
        for row in X:
            scaled_row = [
                (val - self.means[col_idx]) / self.stds[col_idx]
                for col_idx, val in enumerate(row)
            ]
            scaled_matrix.append(scaled_row)
        return scaled_matrix

    def fit_transform(
        self, X: List[List[float]], feature_names: Optional[List[str]] = None
    ) -> List[List[float]]:
        """Fit to X, then transform X."""
        return self.fit(X, feature_names=feature_names).transform(X)
