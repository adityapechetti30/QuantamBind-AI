"""Verification test suite for FastAPI backend endpoints."""

from __future__ import annotations
import json
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from backend.app.main import app


def run_tests():
    print("=" * 70)
    print(" QuantumBind AI: FastAPI Backend Verification Suite")
    print("=" * 70)

    client = TestClient(app)

    # 1. Test GET /health
    print("\n[Endpoint 1] Testing GET /health...")
    resp_health = client.get("/health")
    print(f" Status: {resp_health.status_code}")
    print(f" Payload: {resp_health.json()}")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "ok"
    assert resp_health.json()["service"] == "QuantumBind AI"
    print(" GET /health PASSED")

    # 2. Test GET /api/demo
    print("\n[Endpoint 2] Testing GET /api/demo...")
    resp_demo = client.get("/api/demo")
    print(f" Status: {resp_demo.status_code}")
    demo_items = resp_demo.json()
    print(f" Retrieved {len(demo_items)} demo items.")
    assert resp_demo.status_code == 200
    assert len(demo_items) > 0
    first_item = demo_items[0]
    print(f" First demo item: {first_item['protein_name']} - {first_item['ligand_name']}")
    assert "protein_id" in first_item
    assert "smiles" in first_item
    print(" GET /api/demo PASSED")

    # 3. Test GET /api/model-metrics
    print("\n[Endpoint 3] Testing GET /api/model-metrics...")
    resp_metrics = client.get("/api/model-metrics")
    print(f" Status: {resp_metrics.status_code}")
    metrics_data = resp_metrics.json()
    print(f" Classical XGBoost Metrics: {metrics_data['classical_xgboost']['metrics']}")
    print(f" Quantum QSVC Metrics:     {metrics_data['quantum_qsvc']['metrics']}")
    assert resp_metrics.status_code == 200
    assert metrics_data["classical_xgboost"]["metrics"]["mae"] is not None
    assert metrics_data["quantum_qsvc"]["metrics"]["accuracy"] is not None
    print(" GET /api/model-metrics PASSED")

    # 4. Test GET /api/model-info
    print("\n[Endpoint 4] Testing GET /api/model-info...")
    resp_info = client.get("/api/model-info")
    print(f" Status: {resp_info.status_code}")
    info_data = resp_info.json()
    print(f" Classical Model:      {info_data['classical_model']}")
    print(f" Quantum Model:        {info_data['quantum_model']}")
    print(f" Quantum Feature Map:  {info_data['quantum_feature_map']}")
    print(f" Number of Qubits:     {info_data['num_qubits']}")
    print(f" Quantum Simulator:    {info_data['quantum_simulator']}")
    assert resp_info.status_code == 200
    assert info_data["num_qubits"] == 4
    print(" GET /api/model-info PASSED")

    # 5. Test POST /api/predict with Real Demo Example (Gefitinib / EGFR Kinase)
    print("\n[Endpoint 5] Testing POST /api/predict (Gefitinib demo)...")
    payload = {
        "protein_id": first_item["protein_id"],
        "ligand_name": first_item["ligand_name"],
        "smiles": first_item["smiles"],
    }
    resp_pred = client.post("/api/predict", json=payload)
    print(f" Status: {resp_pred.status_code}")
    pred_data = resp_pred.json()
    print(f" Response:")
    print(f"   Protein ID:             {pred_data['protein_id']}")
    print(f"   Ligand Name:            {pred_data['ligand_name']}")
    print(f"   Classical Pred (pKd):   {pred_data['classical_prediction']}")
    print(f"   Quantum Pred (Strength): {pred_data['quantum_prediction']}")
    print(f"   Binding Site:           {pred_data['binding_site']}")
    print(f"   Is Demo:                {pred_data['is_demo']}")
    print(f"   Confidence:             {pred_data['confidence']}")

    assert resp_pred.status_code == 200
    assert pred_data["classical_prediction"] > 0
    assert pred_data["quantum_prediction"] in ["Strong", "Moderate", "Weak"]
    assert pred_data["confidence"] is None
    print(" POST /api/predict PASSED")

    # 6. Test Error Handling: Invalid SMILES
    print("\n[Endpoint 6] Testing POST /api/predict error handling (Invalid SMILES)...")
    bad_payload = {
        "protein_id": "PROT_TEST",
        "ligand_name": "TestMolecule",
        "smiles": "NotAValidSmilesString!!!((",
    }
    resp_bad = client.post("/api/predict", json=bad_payload)
    print(f" Status: {resp_bad.status_code} (Expected 400)")
    print(f" Detail: {resp_bad.json()['detail']}")
    assert resp_bad.status_code == 400
    print(" Invalid SMILES error handling PASSED")

    # 7. Test Error Handling: Empty Fields
    print("\n[Endpoint 7] Testing POST /api/predict error handling (Empty field)...")
    empty_payload = {
        "protein_id": "   ",
        "ligand_name": "TestMolecule",
        "smiles": "CCO",
    }
    resp_empty = client.post("/api/predict", json=empty_payload)
    print(f" Status: {resp_empty.status_code} (Expected 422)")
    assert resp_empty.status_code == 422
    print(" Empty field error handling PASSED")

    print("\n" + "=" * 70)
    print(" ALL FASTAPI BACKEND TESTS PASSED SUCCESSFULLY! ")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
