"""Prediction and model inference service for QuantumBind AI."""

from __future__ import annotations
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from ml.preprocessing import load_dataset, MolecularFeatureExtractor
from ml.predict import (
    load_model as load_xgb_model,
    predict_binding_affinity,
    categorize_binding_affinity,
    DEFAULT_MODEL_PATH as DEFAULT_XGB_MODEL_PATH,
    DEFAULT_METADATA_PATH as DEFAULT_XGB_META_PATH,
)
from quantum.quantum_model import (
    QuantumBindingClassifier,
    DEFAULT_QUANTUM_MODEL_DIR,
)
from backend.app.utils.smiles_validator import validate_smiles


class PredictionService:
    """Manages pre-loaded classical XGBoost and Qiskit QSVC models for fast inference."""

    def __init__(self, project_root: str = PROJECT_ROOT):
        self.project_root = project_root
        self.demo_data_path = os.path.join(project_root, "data", "demo_dataset.csv")
        self.demo_records: List[Dict[str, Any]] = []

        # Models & Metadata
        self.xgb_model = None
        self.xgb_metadata: Dict[str, Any] = {}
        self.quantum_classifier: Optional[QuantumBindingClassifier] = None
        self.quantum_metadata: Dict[str, Any] = {}
        self.feature_extractor = MolecularFeatureExtractor()

        self._load_demo_data()
        self._load_models()

    def _load_demo_data(self) -> None:
        """Load demo dataset into memory for metadata lookups and demo endpoints."""
        if os.path.exists(self.demo_data_path):
            raw = load_dataset(self.demo_data_path)
            self.demo_records = [dict(r) for r in raw]
        else:
            self.demo_records = []

    def _load_models(self) -> None:
        """Load trained XGBoost and Qiskit QSVC model artifacts from disk."""
        # 1. Load XGBoost
        xgb_path = os.path.join(self.project_root, "ml", "models", "xgboost_model.json")
        xgb_meta_path = os.path.join(self.project_root, "ml", "models", "model_metadata.json")

        if not os.path.exists(xgb_path):
            raise FileNotFoundError(f"XGBoost model file not found at {xgb_path}.")

        self.xgb_model, self.xgb_metadata = load_xgb_model(
            model_path=xgb_path,
            metadata_path=xgb_meta_path,
        )

        # 2. Load Quantum QSVC
        quantum_dir = os.path.join(self.project_root, "quantum", "models")
        qsvc_model_path = os.path.join(quantum_dir, "qsvc_model.joblib")
        qsvc_meta_path = os.path.join(quantum_dir, "quantum_model_metadata.json")

        if not os.path.exists(qsvc_model_path):
            raise FileNotFoundError(f"Quantum QSVC model not found at {qsvc_model_path}.")

        self.quantum_classifier = QuantumBindingClassifier.load(save_dir=quantum_dir)
        if os.path.exists(qsvc_meta_path):
            with open(qsvc_meta_path, "r", encoding="utf-8") as f:
                self.quantum_metadata = json.load(f)

    def get_demo_examples(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return available demo protein-ligand pairs."""
        examples = []
        for r in self.demo_records[:limit]:
            examples.append({
                "protein_id": r.get("protein_id", ""),
                "protein_name": r.get("protein_name", ""),
                "ligand_name": r.get("ligand_name", ""),
                "smiles": r.get("smiles", ""),
                "binding_site": r.get("binding_site", ""),
                "binding_strength": r.get("binding_strength", ""),
                "binding_affinity": float(r.get("binding_affinity", 0.0)),
            })
        return examples

    def get_model_metrics(self) -> Dict[str, Any]:
        """Return actual ground-truth evaluation metrics calculated during training."""
        classical_metrics = self.xgb_metadata.get("test_metrics", {})
        quantum_metrics = self.quantum_metadata.get("evaluation_metrics", {})

        return {
            "classical_xgboost": {
                "model_name": "XGBoost Regressor (Classical Baseline)",
                "metrics": {
                    "mae": classical_metrics.get("mae"),
                    "rmse": classical_metrics.get("rmse"),
                    "r2": classical_metrics.get("r2"),
                },
                "num_train_samples": self.xgb_metadata.get("num_train_samples"),
                "num_test_samples": self.xgb_metadata.get("num_test_samples"),
                "num_features": self.xgb_metadata.get("num_features"),
            },
            "quantum_qsvc": {
                "model_name": "Qiskit QSVC (Quantum ML Prototype)",
                "metrics": {
                    "accuracy": quantum_metrics.get("accuracy"),
                    "precision": quantum_metrics.get("precision"),
                    "recall": quantum_metrics.get("recall"),
                    "f1_score": quantum_metrics.get("f1_score"),
                },
                "num_qubits": self.quantum_metadata.get("num_qubits", 4),
                "quantum_features": self.quantum_metadata.get("quantum_features", []),
            },
            "disclaimer": (
                "Metrics calculated from actual test predictions on 25 demo samples. "
                "Illustrative prototype benchmarks; not clinically validated."
            ),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Return architectural and configuration information for both models."""
        circuit_info = self.quantum_metadata.get("circuit_summary", {})
        return {
            "classical_model": "XGBoost Regression",
            "quantum_model": "Qiskit QSVC (Quantum Support Vector Classifier)",
            "quantum_feature_map": "ZZFeatureMap (reps=1, linear entanglement)",
            "num_qubits": int(self.quantum_metadata.get("num_qubits", 4)),
            "quantum_simulator": "Qiskit Local Statevector Simulator (Offline)",
            "quantum_features": self.quantum_metadata.get("quantum_features", [
                "molecular_weight",
                "logp",
                "h_bond_donors",
                "h_bond_acceptors",
            ]),
            "circuit_summary": circuit_info,
            "disclaimer": (
                "Classical model predicts numerical binding affinity (pKd). "
                "Quantum model classifies prototype binding strength (Strong/Moderate/Weak). "
                "No claim of quantum advantage."
            ),
        }

    def predict(
        self,
        protein_id: str,
        ligand_name: str,
        smiles: str,
    ) -> Dict[str, Any]:
        """Execute hybrid Classical XGBoost + Quantum QSVC inference pipeline."""
        # 1. SMILES Validation
        is_valid, error_msg = validate_smiles(smiles)
        if not is_valid:
            raise ValueError(error_msg)

        # 2. Check for demo dataset match
        matched_demo: Optional[Dict[str, Any]] = None
        for record in self.demo_records:
            if (
                record.get("smiles", "").strip() == smiles.strip()
                or (
                    record.get("protein_id", "").strip().lower() == protein_id.strip().lower()
                    and record.get("ligand_name", "").strip().lower() == ligand_name.strip().lower()
                )
            ):
                matched_demo = record
                break

        is_demo = matched_demo is not None
        binding_site = (
            matched_demo.get("binding_site", "Primary binding pocket")
            if matched_demo
            else "Target binding pocket (in silico estimate)"
        )

        # 3. Feature Extraction
        mol_descriptors = self.feature_extractor.extract_from_smiles(smiles)

        if matched_demo:
            mw = float(matched_demo.get("molecular_weight", 350.0))
            logp = float(matched_demo.get("logp", 2.5))
            hbd = float(matched_demo.get("h_bond_donors", 2.0))
            hba = float(matched_demo.get("h_bond_acceptors", 5.0))
            rot_bonds = float(matched_demo.get("rotatable_bonds", 4.0))
        else:
            mw = float(mol_descriptors.get("heavy_atom_count", 25.0) * 14.0)
            logp = float(mol_descriptors.get("aromatic_ratio", 0.5) * 4.0)
            hbd = float(mol_descriptors.get("nitrogen_count", 2.0))
            hba = float(mol_descriptors.get("oxygen_count", 4.0))
            rot_bonds = float(mol_descriptors.get("branching_index", 3.0))

        # 4. Classical XGBoost Prediction
        table_vals = [mw, logp, hbd, hba, rot_bonds]
        raw_vector = table_vals + [float(v) for v in mol_descriptors.values()]

        means = self.xgb_metadata.get("scaler_means", [])
        stds = self.xgb_metadata.get("scaler_stds", [])
        if means and stds and len(means) == len(raw_vector):
            scaled_vector = [
                (raw_vector[i] - means[i]) / stds[i] for i in range(len(raw_vector))
            ]
        else:
            scaled_vector = raw_vector

        xgb_result = predict_binding_affinity(
            model=self.xgb_model,
            features=scaled_vector,
            feature_names=self.xgb_metadata.get("feature_names"),
        )
        pkd_pred = float(round(xgb_result["predicted_binding_affinity"], 2))
        classical_strength = categorize_binding_affinity(pkd_pred)

        # 5. Quantum QSVC Prediction
        quantum_features_dict = {
            "molecular_weight": mw,
            "logp": logp,
            "h_bond_donors": hbd,
            "h_bond_acceptors": hba,
        }
        quantum_pred = self.quantum_classifier.predict_single(quantum_features_dict)
        quantum_strength = quantum_pred["predicted_binding_strength"]

        return {
            "protein_id": protein_id,
            "ligand_name": ligand_name,
            "binding_affinity_pkd": pkd_pred,
            "binding_strength": quantum_strength,
            "classical_prediction": pkd_pred,
            "quantum_prediction": quantum_strength,
            "classical_strength_estimate": classical_strength,
            "binding_site": binding_site,
            "is_demo": is_demo,
            "confidence": None,
            "molecular_weight": round(mw, 2),
            "logp": round(logp, 2),
            "disclaimer": (
                "XGBoost provides numerical pKd regression. "
                "Qiskit QSVC provides quantum-kernel binding strength classification. "
                "Demo data prototype results for educational and exploration purposes."
            ),
        }


# Singleton instance
_service_instance: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Return initialized singleton PredictionService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PredictionService()
    return _service_instance
