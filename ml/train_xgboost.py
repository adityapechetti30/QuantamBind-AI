"""Training script for XGBoost Classical ML Baseline in QuantumBind AI."""

from __future__ import annotations
import json
import os
import sys

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import xgboost as xgb
from ml.preprocessing import PreprocessingPipeline
from ml.evaluate import calculate_metrics, print_evaluation_summary
from ml.predict import (
    categorize_binding_affinity,
    predict_binding_affinity,
    predict_from_molecular_input,
    load_model,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_METADATA_PATH,
    PROTOTYPE_CATEGORY_DISCLAIMER,
)


def train_and_evaluate(
    dataset_path: str = os.path.join(PROJECT_ROOT, "data", "demo_dataset.csv"),
    model_save_dir: str = DEFAULT_MODEL_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train XGBoost regression baseline, evaluate on test set, and persist artifacts."""
    print("=" * 70)
    print(" QuantumBind AI: XGBoost Classical Baseline Training")
    print("=" * 70)
    print(f"XGBoost Version: {xgb.__version__}")
    print(f"Dataset:         {dataset_path}")

    # 1. Preprocessing Pipeline Execution
    pipeline = PreprocessingPipeline()
    processed_data = pipeline.process_file(
        dataset_path,
        test_size=test_size,
        random_state=random_state,
        scale_features=True,
    )

    X_train = processed_data.X_train
    X_test = processed_data.X_test
    y_train = processed_data.y_train
    y_test = processed_data.y_test
    feature_names = processed_data.feature_names
    scaler = processed_data.scaler

    num_train = len(X_train)
    num_test = len(X_test)
    num_features = len(feature_names)

    print(f"Training samples: {num_train}")
    print(f"Testing samples:  {num_test}")
    print(f"Feature count:    {num_features}")

    # 2. Configure and Train XGBoost (Native booster for robust cross-platform execution)
    # Conservative hyperparameters tailored for the demo dataset
    xgb_params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "alpha": 0.1,
        "lambda": 1.0,
        "seed": random_state,
    }

    X_train_arr = np.array(X_train, dtype=np.float32)
    y_train_arr = np.array(y_train, dtype=np.float32)
    X_test_arr = np.array(X_test, dtype=np.float32)
    y_test_arr = np.array(y_test, dtype=np.float32)

    dtrain = xgb.DMatrix(X_train_arr, label=y_train_arr, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test_arr, label=y_test_arr, feature_names=feature_names)

    print("\nTraining XGBoost Regressor (35 boosting rounds)...")
    bst = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=35,
        evals=[(dtrain, "train")],
        verbose_eval=False,
    )
    print("Training complete.")

    # 3. Predict on Test Set
    preds_raw = bst.predict(dtest)
    y_pred = [float(round(p, 4)) for p in preds_raw]

    # 4. Calculate Ground-Truth Metrics (Calculated strictly from actual predictions)
    metrics = calculate_metrics(y_true=y_test, y_pred=y_pred)
    print_evaluation_summary(metrics, dataset_name="Held-out Test Set", sample_count=num_test)

    # Detailed Predictions Table
    print("\nDetailed Test Set Predictions vs Ground Truth:")
    print(f"{'Idx':<4} | {'Actual pKd':<12} | {'Predicted pKd':<14} | {'Abs Error':<10} | {'Strength':<10}")
    print("-" * 62)
    for i, (actual, pred) in enumerate(zip(y_test, y_pred)):
        err = abs(actual - pred)
        cat = categorize_binding_affinity(pred)
        print(f"{i:<4} | {actual:<12.2f} | {pred:<14.2f} | {err:<10.2f} | {cat:<10}")

    # 5. Save Model & Preprocessing Metadata Locally
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, "xgboost_model.json")
    metadata_path = os.path.join(model_save_dir, "model_metadata.json")

    bst.save_model(model_path)
    metadata = {
        "model_name": "XGBoost Regressor (Classical Baseline)",
        "xgboost_version": str(xgb.__version__),
        "num_train_samples": num_train,
        "num_test_samples": num_test,
        "num_features": num_features,
        "feature_names": feature_names,
        "hyperparameters": xgb_params,
        "scaler_means": scaler.means,
        "scaler_stds": scaler.stds,
        "test_metrics": metrics,
        "model_path": model_path,
        "disclaimer": PROTOTYPE_CATEGORY_DISCLAIMER,
        "dataset_size_notice": (
            "NOTICE: The current dataset contains 25 demo samples. Metrics are calculated "
            "for software workflow verification and do not represent statistically robust "
            "clinical or pharmacological performance."
        ),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel persisted to:    {model_path}")
    print(f"Metadata persisted to: {metadata_path}")

    # 6. Verification: Reload saved model and test sample prediction
    print("\nVerifying saved model reload and inference...")
    reloaded_model, reloaded_meta = load_model(model_path, metadata_path)

    # Test sample A: First sample in the test set
    sample_feat = X_test[0]
    sample_actual = y_test[0]
    sample_result = predict_binding_affinity(
        reloaded_model,
        sample_feat,
        model_name="XGBoost Regressor (Reloaded from Disk)",
        feature_names=feature_names,
    )

    print("Verification Prediction A (Held-out test vector):")
    print(f"  Actual Binding Affinity:    {sample_actual:.2f}")
    print(f"  Predicted Binding Affinity: {sample_result['predicted_binding_affinity']:.2f}")
    print(f"  Prototype Strength:         {sample_result['binding_strength']}")
    print(f"  Model Name:                 {sample_result['model_name']}")

    # Test sample B: End-to-end raw SMILES inference (Gefitinib / EGFR Kinase)
    print("\nVerification Prediction B (Raw SMILES & Chemical Descriptors for Gefitinib):")
    mol_result = predict_from_molecular_input(
        model=reloaded_model,
        scaler_metadata=reloaded_meta,
        smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        molecular_weight=446.90,
        logp=3.75,
        h_bond_donors=1,
        h_bond_acceptors=7,
        rotatable_bonds=7,
    )
    print(f"  Input Molecule:             Gefitinib")
    print(f"  Predicted Binding Affinity: {mol_result['predicted_binding_affinity']:.2f}")
    print(f"  Prototype Strength:         {mol_result['binding_strength']}")
    print(f"  Feature Extractor Used:     {mol_result['feature_extractor']}")

    # Small dataset notice
    print("\n" + "=" * 70)
    print(metadata["dataset_size_notice"])
    print("=" * 70)

    return {
        "xgboost_version": xgb.__version__,
        "num_train_samples": num_train,
        "num_test_samples": num_test,
        "num_features": num_features,
        "metrics": metrics,
        "sample_prediction": sample_result,
        "molecular_prediction": mol_result,
        "model_path": model_path,
        "metadata_path": metadata_path,
    }


if __name__ == "__main__":
    train_and_evaluate()
