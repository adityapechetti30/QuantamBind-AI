"""Data loader and validation module for QuantumBind AI."""

from __future__ import annotations
import csv
import os
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False


REQUIRED_COLUMNS: List[str] = [
    "protein_id",
    "protein_name",
    "ligand_name",
    "smiles",
    "molecular_weight",
    "logp",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "binding_affinity",
    "binding_strength",
    "binding_site",
]

NUMERICAL_COLUMNS: List[str] = [
    "molecular_weight",
    "logp",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "binding_affinity",
]


def load_dataset(filepath: str) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """Load binding dataset from a CSV file.

    Returns a pandas DataFrame if pandas is installed, otherwise a list of row dicts.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    if HAS_PANDAS:
        df = pd.read_csv(filepath)
        return df

    # Fallback loader using standard library csv module
    rows: List[Dict[str, Any]] = []
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row: Dict[str, Any] = dict(row)
            for num_col in NUMERICAL_COLUMNS:
                if num_col in parsed_row and parsed_row[num_col] not in (None, ""):
                    try:
                        parsed_row[num_col] = float(parsed_row[num_col])
                    except ValueError:
                        pass
            rows.append(parsed_row)
    return rows


def validate_dataset(
    data: Union[pd.DataFrame, List[Dict[str, Any]]]
) -> Tuple[bool, List[str]]:
    """Validate schema, required columns, and data consistency.

    Returns:
        (is_valid, validation_errors)
    """
    errors: List[str] = []

    # 1. Determine columns and row count
    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        columns = list(data.columns)
        num_rows = len(data)
    elif isinstance(data, list):
        if not data:
            return False, ["Dataset is empty."]
        columns = list(data[0].keys())
        num_rows = len(data)
    else:
        return False, [f"Unsupported dataset format: {type(data)}"]

    if num_rows == 0:
        return False, ["Dataset contains 0 rows."]

    # 2. Check for missing required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 3. Row-level integrity checks
    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        for col in REQUIRED_COLUMNS:
            if col in data.columns and data[col].isnull().any():
                null_count = int(data[col].isnull().sum())
                errors.append(f"Column '{col}' has {null_count} null/missing values.")

        if "molecular_weight" in data.columns and (data["molecular_weight"] <= 0).any():
            errors.append("Column 'molecular_weight' must contain positive non-zero values.")

        for non_neg in ["h_bond_donors", "h_bond_acceptors", "rotatable_bonds"]:
            if non_neg in data.columns and (data[non_neg] < 0).any():
                errors.append(f"Column '{non_neg}' cannot contain negative values.")
    else:
        for idx, row in enumerate(data):
            for col in REQUIRED_COLUMNS:
                val = row.get(col)
                if val is None or (isinstance(val, str) and val.strip() == ""):
                    errors.append(f"Row {idx}: Missing value for required column '{col}'.")

            mw = row.get("molecular_weight")
            if mw is not None and isinstance(mw, (int, float)) and mw <= 0:
                errors.append(f"Row {idx}: Invalid molecular_weight {mw} <= 0.")

    is_valid = len(errors) == 0
    return is_valid, errors
