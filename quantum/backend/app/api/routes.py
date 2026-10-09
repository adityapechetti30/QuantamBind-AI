"""API route endpoints for QuantumBind AI."""

from __future__ import annotations
import logging
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Depends, status

from backend.app.schemas import (
    HealthResponse,
    DemoItem,
    PredictRequest,
    PredictResponse,
    ModelMetricsResponse,
    ModelInfoResponse,
)
from backend.app.services import PredictionService, get_prediction_service

logger = logging.getLogger("quantumbind.api")

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Return system and service health status."""
    return HealthResponse(status="ok", service="QuantumBind AI")


@router.get("/api/demo", response_model=List[DemoItem], tags=["Data"])
def get_demo_molecules(
    limit: int = 25,
    service: PredictionService = Depends(get_prediction_service),
) -> List[DemoItem]:
    """Retrieve curated demo protein-ligand pairs from the local dataset."""
    try:
        examples = service.get_demo_examples(limit=limit)
        return [DemoItem(**ex) for ex in examples]
    except Exception as exc:
        logger.error(f"Error fetching demo records: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load demo records from dataset.",
        )


@router.post("/api/predict", response_model=PredictResponse, tags=["Inference"])
def predict_protein_ligand_binding(
    payload: PredictRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictResponse:
    """Execute hybrid Classical XGBoost regression + Quantum QSVC classification."""
    try:
        result = service.predict(
            protein_id=payload.protein_id,
            ligand_name=payload.ligand_name,
            smiles=payload.smiles,
        )
        return PredictResponse(**result)
    except ValueError as val_err:
        logger.warning(f"Validation error in /api/predict: {val_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error(f"Inference pipeline execution error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while executing the prediction pipeline. Please verify input data.",
        )


@router.get("/api/model-metrics", response_model=ModelMetricsResponse, tags=["Models"])
def get_model_evaluation_metrics(
    service: PredictionService = Depends(get_prediction_service),
) -> ModelMetricsResponse:
    """Return ground-truth evaluation metrics for XGBoost and Quantum QSVC."""
    try:
        metrics = service.get_model_metrics()
        return ModelMetricsResponse(**metrics)
    except Exception as exc:
        logger.error(f"Error fetching model metrics: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model metrics.",
        )


@router.get("/api/model-info", response_model=ModelInfoResponse, tags=["Models"])
def get_model_architectural_info(
    service: PredictionService = Depends(get_prediction_service),
) -> ModelInfoResponse:
    """Return metadata regarding the classical and quantum model architectures."""
    try:
        info = service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as exc:
        logger.error(f"Error fetching model info: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model architectural specifications.",
        )
