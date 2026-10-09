"""Quantum Machine Learning Classifier for QuantumBind AI using Qiskit QSVC."""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import joblib

import qiskit
import qiskit_machine_learning as qml
from qiskit_machine_learning.algorithms import QSVC

from quantum.quantum_kernel import QuantumBindingKernel
from quantum.quantum_utils import (
    QUANTUM_FEATURE_NAMES,
    CLASS_NAMES,
    QuantumAngleScaler,
    encode_labels,
    decode_labels,
    calculate_classification_metrics,
)

DEFAULT_QUANTUM_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"
)
DEFAULT_METADATA_PATH = os.path.join(
    DEFAULT_QUANTUM_MODEL_DIR, "quantum_model_metadata.json"
)
DEFAULT_QSVC_PATH = os.path.join(DEFAULT_QUANTUM_MODEL_DIR, "qsvc_model.joblib")

QUANTUM_PROTOTYPE_DISCLAIMER = (
    "NOTICE: This model is an exploratory quantum-kernel-based machine learning prototype. "
    "It uses a 4-qubit parameterized ZZFeatureMap to classify binding strength into demo categories "
    "('Strong', 'Moderate', 'Weak'). No claim of quantum advantage or superior clinical accuracy is made."
)


class QuantumBindingClassifier:
    """Quantum Support Vector Classifier (QSVC) utilizing a Qiskit quantum feature map."""

    def __init__(
        self,
        num_qubits: int = 4,
        reps: int = 1,
        entanglement: str = "linear",
        c_param: float = 1.0,
    ):
        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.c_param = c_param

        self.kernel = QuantumBindingKernel(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
        )
        self.scaler = QuantumAngleScaler()
        self.qsvc = QSVC(
            quantum_kernel=self.kernel._fidelity_kernel,
            C=c_param,
        )
        self.is_fitted = False
        self.feature_names = QUANTUM_FEATURE_NAMES

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Union[List[str], np.ndarray],
    ) -> "QuantumBindingClassifier":
        """Train QSVC on normalized quantum features."""
        X_arr = np.asarray(X_train, dtype=np.float64)
        y_enc = encode_labels(y_train)

        # Normalize features to [0, pi] for angle encoding in quantum circuit
        X_scaled = self.scaler.fit_transform(X_arr)

        # Fit QSVC using Qiskit quantum kernel
        self.qsvc.fit(X_scaled, y_enc)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        """Predict binding strength category ('Strong', 'Moderate', 'Weak')."""
        if not self.is_fitted:
            raise RuntimeError("QuantumBindingClassifier must be fitted before predict.")

        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler.transform(X_arr)
        preds_int = self.qsvc.predict(X_scaled)
        return decode_labels(preds_int)

    def predict_single(
        self,
        features: Union[List[float], Dict[str, float]],
    ) -> Dict[str, Any]:
        """Inference helper for a single sample input."""
        if isinstance(features, dict):
            vec = [float(features.get(f, 0.0)) for f in self.feature_names]
        else:
            vec = [float(v) for v in features]

        arr = np.array([vec], dtype=np.float64)
        cat = self.predict(arr)[0]

        return {
            "predicted_binding_strength": cat,
            "model_name": "Quantum Kernel Classifier (Qiskit QSVC)",
            "num_qubits": self.num_qubits,
            "feature_map": "ZZFeatureMap (4-qubit)",
            "simulator": "Qiskit Local Statevector Simulator",
            "disclaimer": QUANTUM_PROTOTYPE_DISCLAIMER,
        }

    def save(
        self,
        save_dir: str = DEFAULT_QUANTUM_MODEL_DIR,
        evaluation_metrics: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, str]:
        """Save trained QSVC model and metadata locally."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted quantum model.")

        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "qsvc_model.joblib")
        metadata_path = os.path.join(save_dir, "quantum_model_metadata.json")

        joblib.dump(self.qsvc, model_path)

        metadata = {
            "model_name": "Quantum Kernel Classifier (Qiskit QSVC)",
            "qiskit_version": str(qiskit.__version__),
            "qiskit_machine_learning_version": str(qml.__version__),
            "num_qubits": self.num_qubits,
            "quantum_features": self.feature_names,
            "circuit_summary": self.kernel.get_circuit_summary(),
            "scaler_mins": self.scaler.feature_mins.tolist() if self.scaler.feature_mins is not None else [],
            "scaler_maxs": self.scaler.feature_maxs.tolist() if self.scaler.feature_maxs is not None else [],
            "classes": CLASS_NAMES,
            "evaluation_metrics": evaluation_metrics or {},
            "disclaimer": QUANTUM_PROTOTYPE_DISCLAIMER,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return model_path, metadata_path

    @classmethod
    def load(
        cls,
        save_dir: str = DEFAULT_QUANTUM_MODEL_DIR,
    ) -> "QuantumBindingClassifier":
        """Load trained QSVC model and metadata from disk."""
        model_path = os.path.join(save_dir, "qsvc_model.joblib")
        metadata_path = os.path.join(save_dir, "quantum_model_metadata.json")

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Quantum model artifacts not found at {save_dir}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        classifier = cls(
            num_qubits=meta.get("num_qubits", 4),
            reps=1,
            entanglement="linear",
        )
        classifier.qsvc = joblib.load(model_path)
        classifier.scaler.feature_mins = np.array(meta["scaler_mins"])
        classifier.scaler.feature_maxs = np.array(meta["scaler_maxs"])
        ranges = classifier.scaler.feature_maxs - classifier.scaler.feature_mins
        ranges[ranges < 1e-9] = 1.0
        classifier.scaler.ranges = ranges
        classifier.scaler.is_fitted = True
        classifier.is_fitted = True
        classifier.feature_names = meta.get("quantum_features", QUANTUM_FEATURE_NAMES)

        return classifier
