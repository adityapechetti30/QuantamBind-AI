"""Molecular feature extraction module for QuantumBind AI.

Provides:
- RDKit-based molecular descriptor calculation when RDKit is installed.
- A deterministic, rule-based SMILES tokenizer fallback extractor when RDKit is not installed.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    Chem = None
    Descriptors = None
    rdMolDescriptors = None
    HAS_RDKIT = False


def has_rdkit() -> bool:
    """Return True if RDKit is available in the current environment."""
    return HAS_RDKIT


class MolecularFeatureExtractor:
    """Extracts chemical features from SMILES strings using RDKit or a documented fallback."""

    def __init__(self, force_fallback: bool = False):
        self.force_fallback = force_fallback
        self.use_rdkit = HAS_RDKIT and not force_fallback

    @property
    def extractor_mode(self) -> str:
        return "RDKit" if self.use_rdkit else "Fallback-SMILES-Tokenizer"

    def extract_from_smiles(self, smiles: str) -> Dict[str, float]:
        """Compute numerical molecular descriptors for a given SMILES string."""
        if self.use_rdkit:
            return self._extract_rdkit(smiles)
        return self._extract_fallback(smiles)

    def _extract_rdkit(self, smiles: str) -> Dict[str, float]:
        """Extract descriptors using RDKit."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Fall back to heuristic parser if SMILES fails RDKit sanitation
            return self._extract_fallback(smiles)

        return {
            "mol_weight": float(Descriptors.MolWt(mol)),
            "mol_logp": float(Descriptors.MolLogP(mol)),
            "num_hbd": float(Descriptors.NumHDonors(mol)),
            "num_hba": float(Descriptors.NumHAcceptors(mol)),
            "num_rotatable_bonds": float(Descriptors.NumRotatableBonds(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
            "ring_count": float(Descriptors.RingCount(mol)),
            "heavy_atom_count": float(Descriptors.HeavyAtomCount(mol)),
            "fraction_csp3": float(Descriptors.FractionCSP3(mol)),
            "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        }

    def _extract_fallback(self, smiles: str) -> Dict[str, float]:
        """Fallback feature extractor when RDKit is not installed.

        Methodology:
        Extracts structural, constitutional, and topological features directly
        from the SMILES specification:
        - Atom counts (C, N, O, S, F, Cl)
        - Aromatic atom density (lowercase organic tokens)
        - Unsaturation index (double '=' and triple '#' bonds)
        - Branching complexity (number of branches indicated by parentheses)
        - Cyclicity (digit occurrences representing ring openings/closures)
        - Heteroatom ratio
        """
        # Normalize and remove charge/bracket decorators for basic atom tally
        cleaned_smiles = smiles.strip()

        # Count Halogens first (multi-letter tokens)
        cl_count = len(re.findall(r"Cl", cleaned_smiles))
        br_count = len(re.findall(r"Br", cleaned_smiles))

        # Single atom counts (avoiding double counting Cl/Br)
        wo_halogens = re.sub(r"Cl|Br", "", cleaned_smiles)
        c_aliphatic = len(re.findall(r"C", wo_halogens))
        c_aromatic = len(re.findall(r"c", wo_halogens))
        n_count = len(re.findall(r"[Nn]", wo_halogens))
        o_count = len(re.findall(r"[Oo]", wo_halogens))
        s_count = len(re.findall(r"[Ss]", wo_halogens))
        f_count = len(re.findall(r"F", wo_halogens))

        total_heavy_atoms = (
            c_aliphatic + c_aromatic + n_count + o_count + s_count + f_count + cl_count + br_count
        )
        if total_heavy_atoms == 0:
            total_heavy_atoms = 1.0

        heteroatom_count = n_count + o_count + s_count + f_count + cl_count + br_count
        heteroatom_ratio = heteroatom_count / total_heavy_atoms

        aromatic_atoms = c_aromatic + len(re.findall(r"[nos]", wo_halogens))
        aromatic_ratio = aromatic_atoms / total_heavy_atoms

        double_bonds = len(re.findall(r"=", cleaned_smiles))
        triple_bonds = len(re.findall(r"#", cleaned_smiles))
        branches = len(re.findall(r"\(", cleaned_smiles))
        ring_closures = len(re.findall(r"[0-9]", cleaned_smiles)) // 2

        # Estimated topological surface proxy
        polarity_proxy = float(n_count * 12.0 + o_count * 9.0 + s_count * 25.0)

        return {
            "smiles_length": float(len(cleaned_smiles)),
            "heavy_atom_count": float(total_heavy_atoms),
            "carbon_count": float(c_aliphatic + c_aromatic),
            "nitrogen_count": float(n_count),
            "oxygen_count": float(o_count),
            "sulfur_count": float(s_count),
            "halogen_count": float(f_count + cl_count + br_count),
            "heteroatom_ratio": round(float(heteroatom_ratio), 4),
            "aromatic_ratio": round(float(aromatic_ratio), 4),
            "double_bonds": float(double_bonds),
            "triple_bonds": float(triple_bonds),
            "branching_index": float(branches),
            "estimated_ring_closures": float(ring_closures),
            "polarity_proxy": round(float(polarity_proxy), 2),
        }
