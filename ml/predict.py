"""Prediction inference module for QuantumBind AI Classical ML."""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xgboost as xgb

from ml.preprocessing import MolecularFeatureExtractor

PROTOTYPE_CATEGORY_DISCLAIMER = (
    "DISCLAIMER: The binding strength categories ('Strong', 'Moderate', 'Weak') "
    "are exploratory demo heuristic thresholds created for hackathon prototyping and "
    "do NOT represent medically or clinically validated pharmacology criteria."
)

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"
)
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "xgboost_model.json")
DEFAULT_METADATA_PATH = os.path.join(DEFAULT_MODEL_DIR, "model_metadata.json")


def categorize_binding_affinity(affinity_score: float) -> str:
    """Convert numerical binding affinity (pK_d) into an exploratory demo category.

    Heuristic thresholds (Demo only):
      - Strong:   affinity >= 7.5  (e.g., sub-micromolar to nanomolar range)
      - Moderate: 6.0 <= affinity < 7.5
      - Weak:     affinity < 6.0

    Note: These thresholds are for hackathon demonstration purposes only.
    """
    if affinity_score >= 7.5:
        return "Strong"
    elif affinity_score >= 6.0:
        return "Moderate"
    else:
        return "Weak"


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
) -> Tuple[xgb.Booster, Dict[str, Any]]:
    """Load the trained XGBoost model and associated preprocessing metadata from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Train the model first via ml/train_xgboost.py."
        )

    model = xgb.Booster()
    model.load_model(model_path)

    metadata: Dict[str, Any] = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, metadata


def predict_binding_affinity(
    model: Any,
    features: Union[List[float], List[List[float]], np.ndarray],
    model_name: str = "XGBoost Regressor (Classical Baseline)",
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Reusable prediction function for pre-extracted/scaled numerical features.

    Args:
        model: Trained XGBoost Booster or Regressor instance.
        features: Scaled feature vector (1D) or matrix (2D).
        model_name: Identifier name for the model.
        feature_names: Optional feature name strings matching model schema.

    Returns:
        Dictionary with predicted binding affinity, prototype strength category,
        and prototype disclaimer.
    """
    # Standardize input to 2D numpy array
    if isinstance(features, np.ndarray):
        arr = features if features.ndim == 2 else features.reshape(1, -1)
        single_input = (features.ndim == 1) or (features.shape[0] == 1)
    elif isinstance(features, list):
        if len(features) > 0 and not isinstance(features[0], list):
            arr = np.array([features], dtype=np.float32)
            single_input = True
        else:
            arr = np.array(features, dtype=np.float32)
            single_input = len(features) == 1
    else:
        arr = np.array(features, dtype=np.float32)
        single_input = True

    # Execute prediction with Booster or sklearn estimator
    if isinstance(model, xgb.Booster):
        fn = feature_names or getattr(model, "feature_names", None)
        dmat = xgb.DMatrix(arr, feature_names=fn)
        preds = model.predict(dmat)
    else:
        preds = model.predict(arr)

    predictions_list = [float(round(p, 4)) for p in preds]

    if single_input:
        pred_val = predictions_list[0]
        category = categorize_binding_affinity(pred_val)
        return {
            "predicted_binding_affinity": pred_val,
            "binding_strength": category,
            "model_name": model_name,
            "disclaimer": PROTOTYPE_CATEGORY_DISCLAIMER,
        }

    return {
        "predictions": [
            {
                "predicted_binding_affinity": p,
                "binding_strength": categorize_binding_affinity(p),
            }
            for p in predictions_list
        ],
        "model_name": model_name,
        "disclaimer": PROTOTYPE_CATEGORY_DISCLAIMER,
    }


def predict_from_molecular_input(
    model: Any,
    scaler_metadata: Dict[str, Any],
    smiles: str,
    molecular_weight: float,
    logp: float,
    h_bond_donors: float,
    h_bond_acceptors: float,
    rotatable_bonds: float,
    model_name: str = "XGBoost Regressor (Classical Baseline)",
) -> Dict[str, Any]:
    """Inference helper accepting raw chemical descriptors and SMILES,

    applies the standard preprocessing and scaling pipeline, and returns predictions.
    """
    extractor = MolecularFeatureExtractor()
    mol_feats = extractor.extract_from_smiles(smiles)

    table_vals = [
        float(molecular_weight),
        float(logp),
        float(h_bond_donors),
        float(h_bond_acceptors),
        float(rotatable_bonds),
    ]
    raw_vector = table_vals + [float(v) for v in mol_feats.values()]

    # Scale using saved scaler statistics
    means = scaler_metadata.get("scaler_means", [])
    stds = scaler_metadata.get("scaler_stds", [])
    if means and stds and len(means) == len(raw_vector):
        scaled_vector = [
            (raw_vector[i] - means[i]) / stds[i] for i in range(len(raw_vector))
        ]
    else:
        scaled_vector = raw_vector

    feature_names = scaler_metadata.get("feature_names", None)
    result = predict_binding_affinity(
        model=model,
        features=scaled_vector,
        model_name=model_name,
        feature_names=feature_names,
    )
    result["input_smiles"] = smiles
    result["feature_extractor"] = extractor.extractor_mode
    return result
