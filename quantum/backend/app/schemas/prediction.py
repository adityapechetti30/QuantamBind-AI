"""Pydantic schemas for QuantumBind AI API."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "QuantumBind AI"


class DemoItem(BaseModel):
    protein_id: str
    protein_name: str
    ligand_name: str
    smiles: str
    binding_site: Optional[str] = None
    binding_strength: Optional[str] = None
    binding_affinity: Optional[float] = None


class PredictRequest(BaseModel):
    protein_id: str = Field(..., description="Target Protein Identifier (e.g., PROT_001 or EGFR)")
    ligand_name: str = Field(..., description="Ligand / Small molecule name")
    smiles: str = Field(..., description="Valid chemical SMILES representation")

    @field_validator("protein_id", "ligand_name", "smiles")
    @classmethod
    def validate_non_empty(cls, v: str, info) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError(f"Field '{info.field_name}' cannot be empty or whitespace.")
        return clean


class PredictResponse(BaseModel):
    protein_id: str
    ligand_name: str
    binding_affinity_pkd: float
    binding_strength: str
    classical_prediction: float
    quantum_prediction: str
    binding_site: str
    is_demo: bool
    confidence: Optional[float] = None
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    disclaimer: Optional[str] = None


class ModelMetricsResponse(BaseModel):
    classical_xgboost: Dict[str, Any]
    quantum_qsvc: Dict[str, Any]
    disclaimer: str


class ModelInfoResponse(BaseModel):
    classical_model: str
    quantum_model: str
    quantum_feature_map: str
    num_qubits: int
    quantum_simulator: str
    quantum_features: List[str]
    disclaimer: str
