# Classical Machine Learning Module

This module contains classical AI/ML pipelines for predicting protein-ligand binding affinities.

## Components
- **`preprocessing/`**: End-to-end dataset loader, schema validation, molecular descriptor extractor (RDKit with SMILES tokenizer fallback), feature scaler, and train/test splitting.
- **`train_xgboost.py`**: Trains the XGBoost regression baseline on the demo dataset, evaluates test metrics, and persists model artifacts.
- **`evaluate.py`**: Calculates ground-truth evaluation metrics: MAE, RMSE, and $R^2$.
- **`predict.py`**: Reusable inference functions and prototype binding strength categorizer (`Strong`, `Moderate`, `Weak`).
- **`models/`**: Local storage for trained model weights (`xgboost_model.json`) and scaler metadata.

## Usage
### 1. Train the Baseline
```bash
python ml/train_xgboost.py
```

### 2. Run Inference
```python
from ml.predict import load_model, predict_from_molecular_input

model, meta = load_model()
res = predict_from_molecular_input(
    model=model,
    scaler_metadata=meta,
    smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    molecular_weight=446.90,
    logp=3.75,
    h_bond_donors=1,
    h_bond_acceptors=7,
    rotatable_bonds=7,
)
print(res["predicted_binding_affinity"], res["binding_strength"])
```

## Important Notice on Dataset Size
> [!NOTE]
> The current demo dataset consists of 25 exploratory sample records (20 train, 5 test). While fully functional for architecture and pipeline verification, performance metrics on this sample size are illustrative and should not be interpreted as statistically validated clinical or pharmacological benchmarks.
