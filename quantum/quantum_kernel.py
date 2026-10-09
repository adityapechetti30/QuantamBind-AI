"""Quantum Kernel implementation for QuantumBind AI using Qiskit."""

from __future__ import annotations
import math
from typing import Any, List, Optional, Tuple, Union
import numpy as np

import qiskit
from qiskit import QuantumCircuit
try:
    from qiskit.circuit.library import zz_feature_map
    HAS_NEW_FEATURE_MAP = True
except ImportError:
    from qiskit.circuit.library import ZZFeatureMap as zz_feature_map
    HAS_NEW_FEATURE_MAP = False

from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_machine_learning.kernels import FidelityQuantumKernel


def build_quantum_feature_map(
    feature_dimension: int = 4,
    reps: int = 1,
    entanglement: str = "linear",
) -> QuantumCircuit:
    """Build a parameterized Qiskit ZZ-Feature Map circuit.

    Maps classical features x in R^n to an n-qubit quantum Hilbert state:
        |psi(x)> = U_{Phi(x)} |0^n>
    where U_{Phi(x)} is composed of Hadamard layers and parameterized
    single-qubit Rz and two-qubit Rzz entangling rotations.
    """
    feature_map = zz_feature_map(
        feature_dimension=feature_dimension,
        reps=reps,
        entanglement=entanglement,
    )
    return feature_map


class QuantumBindingKernel:
    """Computes quantum state transition fidelity kernel matrices using Qiskit.

    Mathematical Definition:
        K(x_i, x_j) = |<psi(x_i) | psi(x_j)>|^2
                     = |<0^n | U^dagger_{Phi(x_j)} U_{Phi(x_i)} | 0^n>|^2
    """

    def __init__(
        self,
        num_qubits: int = 4,
        reps: int = 1,
        entanglement: str = "linear",
    ):
        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.feature_map = build_quantum_feature_map(
            feature_dimension=num_qubits,
            reps=reps,
            entanglement=entanglement,
        )
        # Use Qiskit Machine Learning FidelityQuantumKernel
        self._fidelity_kernel = FidelityQuantumKernel(feature_map=self.feature_map)

    def evaluate(
        self,
        x_vec: np.ndarray,
        y_vec: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate the quantum kernel Gram matrix between datasets.

        Args:
            x_vec: Array of shape (N, num_qubits)
            y_vec: Optional array of shape (M, num_qubits). If None, computes K(x, x).

        Returns:
            Kernel matrix of shape (N, M) with values in [0.0, 1.0].
        """
        x_arr = np.asarray(x_vec, dtype=np.float64)
        y_arr = np.asarray(y_vec, dtype=np.float64) if y_vec is not None else None

        kernel_matrix = self._fidelity_kernel.evaluate(x_vec=x_arr, y_vec=y_arr)
        return np.asarray(kernel_matrix, dtype=np.float64)

    def compute_single_pair_fidelity(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> Tuple[float, QuantumCircuit]:
        """Directly simulate statevectors for two feature vectors and return fidelity.

        Used for verification and circuit inspection.
        """
        c1 = self.feature_map.assign_parameters(np.asarray(x1, dtype=float))
        c2 = self.feature_map.assign_parameters(np.asarray(x2, dtype=float))

        sv1 = Statevector.from_instruction(c1)
        sv2 = Statevector.from_instruction(c2)

        fidelity_val = float(state_fidelity(sv1, sv2))
        return fidelity_val, c1

    def get_circuit_summary(self) -> dict:
        """Return circuit depth, qubit count, and gate operations."""
        return {
            "num_qubits": self.feature_map.num_qubits,
            "depth": self.feature_map.depth(),
            "num_parameters": len(self.feature_map.parameters),
            "gate_counts": dict(self.feature_map.count_ops()),
        }
