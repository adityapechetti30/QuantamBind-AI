# QuantumBind AI Architecture & Methodology

## 1. System Overview

QuantumBind AI implements a hybrid classical-quantum machine learning pipeline for protein-ligand binding analysis:
- **XGBoost Classical Model**: Responsible for continuous regression predicting binding affinity ($pK_d$).
- **Qiskit Quantum Machine Learning Model**: Responsible for discrete binding strength classification (`Strong`, `Moderate`, `Weak`) via quantum kernel methods.

```text
                               Input (Ligand SMILES + Protein Data)
                                               │
                                               ▼
                              Preprocessing & Feature Engineering
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
             Full Feature Set (19D)                          Selected Subset (4D)
                       │                                               │
               Feature Scaling                               Angle Scaling [0, π]
                       │                                               │
                       ▼                                               ▼
              XGBoost Regressor                              ZZFeatureMap (4 Qubits)
                       │                                               │
                       ▼                                               ▼
             Predicted pKd (e.g. 8.48)                      Quantum Fidelity Kernel
                                                                       │
                                                                       ▼
                                                             QSVC Quantum Classifier
                                                                       │
                                                                       ▼
                                                            Binding Strength Category
                                                           ('Strong', 'Moderate', 'Weak')
```

---

## 2. Quantum Machine Learning Component

### A. Why Quantum ML is Being Explored
Drug discovery and molecular docking involve complex, non-linear interactions in high-dimensional conformational space. Quantum machine learning is explored to investigate whether mapping classical molecular descriptors into quantum Hilbert spaces can expose non-linear correlations or structural similarities that are difficult for classical linear kernels to capture.

### B. What a Quantum Feature Map Does
The quantum feature map $\Phi(x)$ is a parameterized quantum circuit that encodes classical normalized features $x = [x_1, x_2, x_3, x_4]^T \in [0, \pi]^4$ into a quantum state:
$$|\psi(x)\rangle = U_{\Phi(x)} |0^{\otimes n}\rangle$$

We employ a **`ZZFeatureMap`** with $n=4$ qubits, $1$ repetition, and linear entanglement. The circuit applies:
1. **Hadamard Gates ($H$)**: Create an equal superposition across all basis states.
2. **Single-Qubit Phase Gates ($R_z(2x_i)$)**: Encode individual feature values into relative quantum phases.
3. **Entangling Two-Qubit Phase Gates ($R_{zz}(2(\pi - x_i)(\pi - x_j))$)**: Introduce pairwise quantum entanglement via CNOT-phase-CNOT ladders, capturing non-linear feature interactions.

### C. What a Quantum Kernel Does
A quantum kernel computes the inner product (transition fidelity) between quantum states prepared by the feature map:
$$K(x_i, x_j) = |\langle \psi(x_i) | \psi(x_j) \rangle|^2 = |\langle 0^{\otimes n} | U^\dagger_{\Phi(x_j)} U_{\Phi(x_i)} | 0^{\otimes n} \rangle|^2$$

This produces a symmetric Gram matrix $K \in \mathbb{R}^{N \times N}$ where each entry $K_{ij} \in [0.0, 1.0]$ represents the quantum overlap between molecule $i$ and molecule $j$.

### D. Why QSVC (Quantum Support Vector Classifier) is Being Used
The Quantum Support Vector Classifier replaces the conventional classical kernel (such as RBF or polynomial) with the Quantum Fidelity Kernel evaluated by Qiskit. The dual optimization problem finds the maximum-margin hyperplane in the quantum Hilbert space without requiring explicit coordinates in the exponentially large state space:
$$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

### E. Quantum Configuration Summary
- **Algorithm**: Quantum Support Vector Classifier (QSVC) via `qiskit_machine_learning.algorithms.QSVC`
- **Feature Map**: `ZZFeatureMap` (4 qubits, reps=1, linear entanglement)
- **Number of Qubits**: 4
- **Number of Quantum Features**: 4 (`molecular_weight`, `logp`, `h_bond_donors`, `h_bond_acceptors`)
- **Simulator**: Local Qiskit Statevector Simulator (offline, zero cloud dependencies, exact state fidelity)
- **Target Classes**: 3 Prototype Categories (`Weak`, `Moderate`, `Strong`)

---

## 3. Scientific Honesty & Limitations of the Prototype

> [!CAUTION]
> **No Claim of Quantum Advantage**: This system is strictly an exploratory quantum-kernel-based machine learning prototype designed for hackathon demonstration. We make no claim that this quantum prototype outperforms XGBoost or that "quantum advantage" has been achieved.

### Key Limitations
1. **Sample Size**: The demo dataset contains 25 curated samples ($20$ training, $5$ testing). While adequate for software architecture and algorithmic verification, it is too small for statistically definitive pharmacological benchmarking.
2. **Feature Dimensionality**: The quantum feature map uses 4 features due to qubit simulation scaling constraints. Classical XGBoost utilizes the full 19-dimensional molecular descriptor set.
3. **Simulation vs. QPU Noise**: The prototype executes on a local statevector simulator assuming ideal, noiseless qubits. Real NISQ quantum computers experience gate infidelities and decoherence.
