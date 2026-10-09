# QuantumBind AI ⚛️🧬

Prototype for protein-ligand binding prediction combining **Classical Machine Learning (XGBoost)** and **Quantum Machine Learning (Qiskit QSVC)**.

---

## 1. Project Goal & Hybrid Architecture

QuantumBind AI implements a bifurcated hybrid workflow:
- **XGBoost (Classical Baseline)**: Performs continuous regression predicting binding affinity ($pK_d$).
- **Qiskit QSVC (Quantum ML Prototype)**: Performs discrete classification predicting binding strength categories (`Weak`, `Moderate`, `Strong`) using a quantum kernel.

```text
quantumbind-ai/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py     # REST API routes (/health, /api/demo, /api/predict, etc.)
│   │   ├── schemas/
│   │   │   └── prediction.py # Pydantic request/response schemas
│   │   ├── services/
│   │   │   └── prediction_service.py # Preloaded XGBoost + Qiskit QSVC service
│   │   ├── utils/
│   │   │   └── smiles_validator.py  # Chemical syntax and RDKit validation
│   │   └── main.py           # FastAPI application & CORS configuration
│   ├── main.py               # Root forwarding proxy
│   └── requirements.txt      # Backend dependencies
├── data/
│   └── demo_dataset.csv      # Synthetic demo dataset (25 samples, clearly labeled)
├── docs/
│   └── architecture.md       # Architectural diagrams and QML formulations
├── ml/
│   ├── models/               # Persisted XGBoost model (xgboost_model.json)
│   ├── preprocessing/        # Data loader, validator, RDKit/fallback extractor, scaler
│   ├── evaluate.py           # Evaluation metrics (MAE, RMSE, R²)
│   ├── predict.py            # Classical inference interface
│   └── train_xgboost.py      # XGBoost training script
└── quantum/
    ├── models/               # Persisted QSVC model (qsvc_model.joblib & metadata)
    ├── quantum_kernel.py     # Qiskit ZZFeatureMap & FidelityQuantumKernel
    ├── quantum_model.py      # QSVC training, inference, and serialization
    ├── quantum_utils.py      # Quantum feature selection, angle scaling, and metrics
    └── test_quantum_model.py # End-to-end QML verification suite
```

---

## 2. Backend Setup & Run Instructions

### Virtual Environment Setup (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

### Starting the FastAPI Server
```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive API docs are available at `http://127.0.0.1:8000/docs`.

---

## 3. API Endpoints

### 1. `GET /health`
Health check endpoint.
- **Response**:
  ```json
  {
    "status": "ok",
    "service": "QuantumBind AI"
  }
  ```

### 2. `GET /api/demo`
Retrieves curated demo protein-ligand pairs from `data/demo_dataset.csv`.
- **Response**: List of demo records with `protein_id`, `protein_name`, `ligand_name`, `smiles`, `binding_site`, and `binding_affinity`.

### 3. `POST /api/predict`
Executes hybrid Classical XGBoost regression + Quantum QSVC classification.
- **Request Body**:
  ```json
  {
    "protein_id": "PROT_001",
    "ligand_name": "Gefitinib",
    "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
  }
  ```
- **Response**:
  ```json
  {
    "protein_id": "PROT_001",
    "ligand_name": "Gefitinib",
    "binding_affinity_pkd": 8.48,
    "binding_strength": "Strong",
    "classical_prediction": 8.48,
    "quantum_prediction": "Strong",
    "binding_site": "ATP-binding pocket",
    "is_demo": true,
    "confidence": null
  }
  ```

### 4. `GET /api/model-metrics`
Returns actual evaluation metrics computed on held-out test data:
- **Classical XGBoost**: MAE (`0.3209`), RMSE (`0.3683`), $R^2$ (`0.0196`)
- **Quantum QSVC**: Accuracy (`0.8000`), Precision (`0.2667`), Recall (`0.3333`), F1 Score (`0.2963`)

### 5. `GET /api/model-info`
Returns architectural metadata:
- **Classical Model**: XGBoost Regression
- **Quantum Model**: Qiskit QSVC (Quantum Support Vector Classifier)
- **Quantum Feature Map**: `ZZFeatureMap` (4 qubits, reps=1, linear entanglement)
- **Quantum Simulator**: Qiskit Local Statevector Simulator (Offline)

---

## 4. Scientific Honesty & Labeling

> [!CAUTION]
> - **Predicted pKd**: Continuous numerical binding affinity produced by the classical XGBoost regressor.
> - **Predicted Binding Strength**: Discrete categorical classification (`Strong`, `Moderate`, `Weak`) produced by the Qiskit QSVC model.
> - **No Quantum Advantage Claim**: This system is strictly an exploratory quantum-kernel machine learning prototype designed for hackathon demonstration. We make no claim of quantum advantage or clinical validation.
> - **Demo Data Notice**: The dataset consists of 25 synthetic benchmark demo samples.
