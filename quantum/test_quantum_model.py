"""Verification and training script for QuantumBind AI Quantum ML Component."""

from __future__ import annotations
import os
import sys

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import qiskit
import qiskit_machine_learning as qml

from ml.preprocessing import load_dataset
from quantum.quantum_utils import (
    QUANTUM_FEATURE_NAMES,
    CLASS_NAMES,
    select_quantum_features,
    encode_labels,
    decode_labels,
    calculate_classification_metrics,
)
from quantum.quantum_kernel import QuantumBindingKernel, build_quantum_feature_map
from quantum.quantum_model import QuantumBindingClassifier, DEFAULT_QUANTUM_MODEL_DIR


def run_quantum_pipeline():
    print("=" * 70)
    print(" QuantumBind AI: Quantum Machine Learning (Qiskit QSVC)")
    print("=" * 70)

    # 1. Version Check
    print(f"Qiskit Version:                  {qiskit.__version__}")
    print(f"Qiskit Machine Learning Version: {qml.__version__}")
    print("Quantum Algorithm:               Quantum Support Vector Classifier (QSVC)")
    print("Simulator:                       Qiskit Local Statevector Simulator (No cloud needed)")

    # 2. Dataset Ingestion & Feature Selection
    dataset_path = os.path.join(PROJECT_ROOT, "data", "demo_dataset.csv")
    records = load_dataset(dataset_path)

    # Train/Test split matching random_state=42 (20 train, 5 test)
    import random
    indices = list(range(len(records)))
    rng = random.Random(42)
    rng.shuffle(indices)

    split_idx = int(len(indices) * 0.8)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]

    X_train_raw, feat_names = select_quantum_features(train_records)
    y_train_labels = [r["binding_strength"] for r in train_records]

    X_test_raw, _ = select_quantum_features(test_records)
    y_test_labels = [r["binding_strength"] for r in test_records]

    print(f"\nTraining samples:                {len(X_train_raw)}")
    print(f"Testing samples:                 {len(X_test_raw)}")
    print(f"Quantum features selected ({len(feat_names)}):     {feat_names}")

    # 3. Quantum Circuit & Kernel Inspection
    print("\n[Step 1] Constructing Quantum Feature Map Circuit...")
    kernel = QuantumBindingKernel(num_qubits=4, reps=1, entanglement="linear")
    circ_summary = kernel.get_circuit_summary()
    print(f" Feature Map:                    ZZFeatureMap (reps=1, entanglement='linear')")
    print(f" Number of Qubits:               {circ_summary['num_qubits']}")
    print(f" Circuit Depth:                  {circ_summary['depth']}")
    print(f" Parameterized Rotations:        {circ_summary['num_parameters']}")
    print(f" Gate Composition:               {circ_summary['gate_counts']}")

    # 4. Verify Single-Pair Quantum State Transition Fidelity
    print("\n[Step 2] Executing Quantum Circuit on Local Simulator...")
    norm_x1 = np.array([0.5, 1.2, 0.8, 2.1])
    norm_x2 = np.array([0.6, 1.1, 0.9, 2.0])
    fidelity_val, bound_circuit = kernel.compute_single_pair_fidelity(norm_x1, norm_x2)
    print(f" Computed Quantum Fidelity |<psi(x1)|psi(x2)>|^2 = {fidelity_val:.4f}")
    assert 0.0 <= fidelity_val <= 1.0, "Quantum fidelity must lie within [0.0, 1.0]"
    print(" Quantum circuit statevector execution verified successfully!")

    # 5. Train QSVC Classifier
    print("\n[Step 3] Training Quantum Support Vector Classifier (QSVC)...")
    classifier = QuantumBindingClassifier(num_qubits=4, reps=1, entanglement="linear", c_param=1.0)
    classifier.fit(X_train_raw, y_train_labels)
    print(" QSVC training complete using Qiskit FidelityQuantumKernel.")

    # 6. Evaluate on Held-out Test Set
    print("\n[Step 4] Evaluating Quantum Classifier on Test Set...")
    preds = classifier.predict(X_test_raw)
    y_test_enc = encode_labels(y_test_labels)
    preds_enc = encode_labels(preds)

    metrics = calculate_classification_metrics(y_test_enc, preds_enc, num_classes=3)
    print("-" * 50)
    print(" Quantum Classifier Test Metrics:")
    print("-" * 50)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print("-" * 50)

    y_test_canonical = decode_labels(y_test_enc)
    print("\nDetailed Test Predictions:")
    print(f"{'Idx':<4} | {'Actual Class':<14} | {'Predicted Class':<16} | {'Match':<6}")
    print("-" * 46)
    for i, (act, pr) in enumerate(zip(y_test_canonical, preds)):
        match = "YES" if act == pr else "NO"
        print(f"{i:<4} | {act:<14} | {pr:<16} | {match:<6}")

    # 7. Model Persistence & Reload Verification
    print("\n[Step 5] Persisting Quantum Model Artifacts...")
    model_path, meta_path = classifier.save(DEFAULT_QUANTUM_MODEL_DIR, evaluation_metrics=metrics)
    print(f" Saved QSVC model:    {model_path}")
    print(f" Saved metadata:      {meta_path}")

    print("\n[Step 6] Testing Model Reload and Single-Sample Prediction...")
    reloaded_clf = QuantumBindingClassifier.load(DEFAULT_QUANTUM_MODEL_DIR)

    # Example 1: Gefitinib (Known high-affinity binder)
    gefitinib_feats = {
        "molecular_weight": 446.90,
        "logp": 3.75,
        "h_bond_donors": 1,
        "h_bond_acceptors": 7,
    }
    pred_gef = reloaded_clf.predict_single(gefitinib_feats)
    print(" Example Prediction 1 (Gefitinib):")
    print(f"   Predicted Binding Strength: {pred_gef['predicted_binding_strength']}")
    print(f"   Model:                     {pred_gef['model_name']}")

    # Example 2: Aspirin (Known weak-affinity binder)
    aspirin_feats = {
        "molecular_weight": 180.16,
        "logp": 1.19,
        "h_bond_donors": 1,
        "h_bond_acceptors": 3,
    }
    pred_asp = reloaded_clf.predict_single(aspirin_feats)
    print(" Example Prediction 2 (Aspirin):")
    print(f"   Predicted Binding Strength: {pred_asp['predicted_binding_strength']}")
    print(f"   Model:                     {pred_asp['model_name']}")

    print("\n" + "=" * 70)
    print(" ALL QUANTUM MACHINE LEARNING TESTS COMPLETED SUCCESSFULLY! ")
    print("=" * 70)

    return {
        "qiskit_version": qiskit.__version__,
        "qiskit_machine_learning_version": qml.__version__,
        "algorithm": "QSVC (Quantum Support Vector Classifier)",
        "feature_map": "ZZFeatureMap (4 qubits, reps=1, linear entanglement)",
        "num_qubits": 4,
        "quantum_features": feat_names,
        "simulator": "Qiskit Local Statevector Simulator",
        "num_train": len(X_train_raw),
        "num_test": len(X_test_raw),
        "metrics": metrics,
        "example_prediction": pred_gef,
    }


if __name__ == "__main__":
    run_quantum_pipeline()
