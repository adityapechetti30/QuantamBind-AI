"""FastAPI main application entry point for QuantumBind AI."""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import router
from backend.app.services import get_prediction_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("quantumbind.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: preload models during application startup."""
    logger.info("Initializing QuantumBind AI Backend Service...")
    try:
        service = get_prediction_service()
        logger.info(
            f"Preloaded XGBoost Regressor and Qiskit QSVC Quantum Classifier successfully. "
            f"Demo records in database: {len(service.demo_records)}"
        )
    except Exception as exc:
        logger.error(f"Critical error during model startup initialization: {exc}")
        raise
    yield
    logger.info("Shutting down QuantumBind AI Backend Service.")


app = FastAPI(
    title="QuantumBind AI API",
    description="Hybrid Classical AI & Quantum Machine Learning Backend for Protein-Ligand Binding Prediction",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for local development (including Vite React frontend)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure internal errors do not expose Python tracebacks to clients."""
    logger.error(f"Unhandled server error on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please verify your request."},
    )


@app.get("/", tags=["System"])
async def root():
    """Root entrypoint providing service status and quick links."""
    return {
        "service": "QuantumBind AI API",
        "status": "online",
        "description": "Hybrid Classical AI (XGBoost) & Quantum Machine Learning (Qiskit QSVC) Service",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "health": "/health",
            "demo_molecules": "/api/demo",
            "predict": "/api/predict",
            "model_metrics": "/api/model-metrics",
            "model_info": "/api/model-info",
            "swagger_docs": "/docs",
            "redoc": "/redoc",
        },
    }


# Mount API routes
app.include_router(router)
