# Quantum Machine Learning Module

This module contains Quantum Machine Learning (QML) implementations built using **Qiskit** for molecular property and binding affinity analysis.

## Key Components
- **Feature Encoding & Quantum Feature Maps**: Map classical molecular descriptors/fingerprints into quantum Hilbert space (e.g., `ZZFeatureMap`, `PauliFeatureMap`, Angle Encoding).
- **Variational Quantum Circuits (VQC / QNN)**: Parameterized quantum circuits trained to predict affinity or classify active vs. inactive binders.
- **Quantum Kernel Methods**: Quantum Kernel Alignment and Quantum Support Vector Machines (QSVM) / Quantum Kernel Ridge Regression (QKRR).
- **Simulators & Backend Execution**: Uses `qiskit-aer` for local noiseless/noisy simulation, with optional support for IBM Quantum hardware.
