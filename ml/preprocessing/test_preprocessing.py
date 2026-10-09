"""Verification and test suite for the QuantumBind AI Preprocessing module."""

import os
import sys

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing import (
    load_dataset,
    validate_dataset,
    MolecularFeatureExtractor,
    has_rdkit,
    FeatureScaler,
    PreprocessingPipeline,
)


def run_tests():
    dataset_path = os.path.join(PROJECT_ROOT, "data", "demo_dataset.csv")
    print("=" * 70)
    print(" QuantumBind AI: Preprocessing Verification Suite")
    print("=" * 70)
    print(f"Dataset path: {dataset_path}")
    print(f"RDKit detected: {has_rdkit()}")

    # 1. Test CSV loading
    print("\n[Step 1] Testing CSV Loading...")
    raw_data = load_dataset(dataset_path)
    count = len(raw_data)
    print(f" Loaded {count} rows successfully.")
    assert count == 25, f"Expected 25 demo rows, got {count}"

    # 2. Test Validation
    print("\n[Step 2] Testing Schema & Data Validation...")
    is_valid, errors = validate_dataset(raw_data)
    print(f" Validation status: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        print(f" Errors: {errors}")
    assert is_valid, "Validation failed on valid demo_dataset.csv!"

    # 2b. Test Validation Error Detection
    corrupted_data = [
        {"protein_id": "P1", "molecular_weight": -10.0}  # missing cols and negative MW
    ]
    is_corr_valid, corr_errors = validate_dataset(corrupted_data)
    assert not is_corr_valid, "Validator should catch missing columns and invalid MW"
    print(f" Intentional corruption check passed: Caught {len(corr_errors)} expected error(s).")

    # 3. Test Feature Extractor (Fallback Mode)
    print("\n[Step 3] Testing Molecular Feature Extractor...")
    fallback_extractor = MolecularFeatureExtractor(force_fallback=True)
    sample_smiles = "CC(=O)Nc1ccc(O)cc1"  # Paracetamol
    feats_fb = fallback_extractor.extract_from_smiles(sample_smiles)
    print(f" Fallback extractor on Acetaminophen ({sample_smiles}):")
    for k, v in feats_fb.items():
        print(f"   - {k}: {v}")
    assert feats_fb["smiles_length"] == len(sample_smiles)
    assert feats_fb["oxygen_count"] == 2.0
    assert feats_fb["nitrogen_count"] == 1.0

    # 4. Test Feature Scaler
    print("\n[Step 4] Testing Feature Scaler (StandardScaler)...")
    dummy_matrix = [
        [10.0, 1.0],
        [20.0, 2.0],
        [30.0, 3.0],
    ]
    scaler = FeatureScaler()
    scaled_dummy = scaler.fit_transform(dummy_matrix)
    col0_vals = [row[0] for row in scaled_dummy]
    col0_mean = sum(col0_vals) / len(col0_vals)
    print(f" Dummy column mean after scaling: {col0_mean:.6f} (expected ~0.0)")
    assert abs(col0_mean) < 1e-5, "Scaled feature mean should be ~0"

    # 5. Test Full Pipeline Execution
    print("\n[Step 5] Testing End-to-End Preprocessing Pipeline...")
    pipeline = PreprocessingPipeline()
    processed = pipeline.process_file(
        dataset_path,
        test_size=0.2,
        random_state=42,
        scale_features=True,
    )

    print(f" Extractor mode utilized: {processed.metadata['feature_extractor_mode']}")
    print(f" Total samples: {processed.metadata['total_samples']}")
    print(f" Train samples: {processed.metadata['train_samples']}")
    print(f" Test samples:  {processed.metadata['test_samples']}")
    print(f" Number of features: {processed.metadata['num_features']}")
    print(f" Feature names ({len(processed.feature_names)}):")
    for i, name in enumerate(processed.feature_names):
        print(f"   [{i:02d}] {name}")

    print("\n First scaled train sample features:")
    print(f"   X_train[0] = {[round(x, 4) for x in processed.X_train[0]]}")
    print(f"   y_train[0] (affinity) = {processed.y_train[0]}")
    print(f"   y_train_class[0] = {processed.y_train_class[0]}")

    # Assertions on pipeline output
    assert len(processed.X_train) == 20, f"Expected 20 train samples, got {len(processed.X_train)}"
    assert len(processed.X_test) == 5, f"Expected 5 test samples, got {len(processed.X_test)}"
    assert len(processed.y_train) == 20
    assert len(processed.y_test) == 5
    assert len(processed.X_train[0]) == processed.metadata["num_features"]

    print("\n" + "=" * 70)
    print(" ALL PREPROCESSING TESTS PASSED SUCCESSFULLY! ")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
