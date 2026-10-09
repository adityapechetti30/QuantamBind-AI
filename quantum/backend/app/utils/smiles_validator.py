"""SMILES syntax and chemical validity checking for QuantumBind AI."""

from __future__ import annotations
import re
from typing import Tuple

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    Chem = None
    HAS_RDKIT = False


def validate_smiles(smiles: str) -> Tuple[bool, str]:
    """Validate SMILES string using RDKit if installed, or syntax heuristics if not.

    Returns:
        (is_valid: bool, error_message: str)
    """
    clean_smiles = smiles.strip()
    if not clean_smiles:
        return False, "SMILES string cannot be empty."

    if HAS_RDKIT and Chem is not None:
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            return False, f"Invalid chemical SMILES: RDKit failed to parse or sanitize '{clean_smiles}'."
        return True, ""

    # Fallback syntactic validation
    # 1. Check balanced parentheses
    if clean_smiles.count("(") != clean_smiles.count(")"):
        return False, "Invalid SMILES syntax: Unbalanced branch parentheses '(' and ')'."

    # 2. Check balanced square brackets
    if clean_smiles.count("[") != clean_smiles.count("]"):
        return False, "Invalid SMILES syntax: Unbalanced square brackets '[' and ']'."

    # 3. Check for valid character set in SMILES
    valid_smiles_chars = r"^[A-Za-z0-9@+\-\[\]\(\)\\\/%=#.:]+$"
    if not re.match(valid_smiles_chars, clean_smiles):
        return False, "Invalid SMILES syntax: Contains unsupported non-chemical characters."

    # 4. Must contain at least one recognized organic atom
    if not re.search(r"[CNOFPSBcnopsb]", clean_smiles):
        return False, "Invalid SMILES: No valid organic/aromatic atoms found."

    return True, ""
