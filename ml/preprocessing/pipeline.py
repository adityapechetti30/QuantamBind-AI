"""End-to-end preprocessing pipeline for QuantumBind AI."""

from __future__ import annotations
from dataclasses import dataclass, field
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from .data_loader import load_dataset, validate_dataset
from .molecular_features import MolecularFeatureExtractor
from .scaler import FeatureScaler

try:
    from sklearn.model_selection import train_test_split
    # Test that train_test_split actually functions without DLL blockage
    _ = train_test_split([0, 1], test_size=0.5, random_state=42)
    HAS_SKLEARN_SPLIT = True
except (ImportError, Exception):
    train_test_split = None
    HAS_SKLEARN_SPLIT = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False


STRENGTH_MAPPING = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}


@dataclass
class PreprocessedData:
    """Container for preprocessed train/test sets and pipeline artifacts."""
    X_train: List[List[float]]
    X_test: List[List[float]]
    y_train: List[float]
    y_test: List[float]
    y_train_class: List[int]
    y_test_class: List[int]
    feature_names: List[str]
    scaler: FeatureScaler
    metadata: Dict[str, Any] = field(default_factory=dict)


class PreprocessingPipeline:
    """Coordinates data ingestion, validation, feature extraction, scaling, and splitting."""

    def __init__(self, force_fallback_extractor: bool = False):
        self.feature_extractor = MolecularFeatureExtractor(force_fallback=force_fallback_extractor)
        self.scaler = FeatureScaler()

    def process_file(
        self,
        filepath: str,
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True,
    ) -> PreprocessedData:
        """Execute full preprocessing pipeline from CSV file."""
        # 1. Load dataset
        raw_data = load_dataset(filepath)

        # 2. Validate schema and constraints
        is_valid, errors = validate_dataset(raw_data)
        if not is_valid:
            raise ValueError(f"Dataset validation failed: {'; '.join(errors)}")

        # Convert to uniform list of dict records
        if HAS_PANDAS and isinstance(raw_data, pd.DataFrame):
            records: List[Dict[str, Any]] = raw_data.to_dict(orient="records")
        else:
            records = raw_data

        # 3. Numerical Feature Preparation
        X_raw: List[List[float]] = []
        y_affinity: List[float] = []
        y_class: List[int] = []
        feature_names: List[str] = []

        table_features = [
            "molecular_weight",
            "logp",
            "h_bond_donors",
            "h_bond_acceptors",
            "rotatable_bonds",
        ]

        for idx, row in enumerate(records):
            smiles = str(row["smiles"])
            mol_feats = self.feature_extractor.extract_from_smiles(smiles)

            # Build feature header order on first row
            if idx == 0:
                feature_names = [f"table_{f}" for f in table_features] + list(mol_feats.keys())

            feature_vector: List[float] = []
            for tf in table_features:
                feature_vector.append(float(row[tf]))
            for k in mol_feats:
                feature_vector.append(float(mol_feats[k]))

            X_raw.append(feature_vector)
            y_affinity.append(float(row["binding_affinity"]))

            strength_label = str(row.get("binding_strength", "Medium")).strip()
            y_class.append(STRENGTH_MAPPING.get(strength_label, 1))

        # 4. Train / Test Split
        if HAS_SKLEARN_SPLIT and train_test_split is not None:
            X_tr, X_te, y_tr, y_te, y_c_tr, y_c_te = train_test_split(
                X_raw,
                y_affinity,
                y_class,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
            )
        else:
            # Deterministic pseudo-random split fallback
            indices = list(range(len(X_raw)))
            rng = random.Random(random_state)
            rng.shuffle(indices)

            split_idx = int(len(indices) * (1.0 - test_size))
            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]

            X_tr = [X_raw[i] for i in train_idx]
            X_te = [X_raw[i] for i in test_idx]
            y_tr = [y_affinity[i] for i in train_idx]
            y_te = [y_affinity[i] for i in test_idx]
            y_c_tr = [y_class[i] for i in train_idx]
            y_c_te = [y_class[i] for i in test_idx]

        # 5. Feature Scaling (fit on train, transform on both)
        if scale_features:
            self.scaler.fit(X_tr, feature_names=feature_names)
            X_tr_final = self.scaler.transform(X_tr)
            X_te_final = self.scaler.transform(X_te)
        else:
            X_tr_final = X_tr
            X_te_final = X_te

        metadata = {
            "total_samples": len(records),
            "train_samples": len(X_tr_final),
            "test_samples": len(X_te_final),
            "num_features": len(feature_names),
            "feature_extractor_mode": self.feature_extractor.extractor_mode,
            "scaled": scale_features,
            "random_state": random_state,
            "test_size": test_size,
        }

        return PreprocessedData(
            X_train=X_tr_final,
            X_test=X_te_final,
            y_train=y_tr,
            y_test=y_te,
            y_train_class=y_c_tr,
            y_test_class=y_c_te,
            feature_names=feature_names,
            scaler=self.scaler,
            metadata=metadata,
        )
