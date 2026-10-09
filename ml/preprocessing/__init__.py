"""QuantumBind AI Preprocessing Module

Provides dataset loading, schema validation, molecular descriptor extraction
(with RDKit and deterministic fallback), numerical feature scaling, and train/test splits.
"""

from .data_loader import load_dataset, validate_dataset
from .molecular_features import (
    MolecularFeatureExtractor,
    has_rdkit,
)
from .scaler import FeatureScaler
from .pipeline import PreprocessingPipeline, PreprocessedData

__all__ = [
    "load_dataset",
    "validate_dataset",
    "MolecularFeatureExtractor",
    "has_rdkit",
    "FeatureScaler",
    "PreprocessingPipeline",
    "PreprocessedData",
]
