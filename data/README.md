# Data Directory

This directory manages datasets for protein-ligand pairs and binding metrics.

## Structure
- `raw/`: Unmodified input data (e.g. CSVs from BindingDB, PDB files, SDF files, raw SMILES strings). Git-ignored except `.gitkeep`.
- `processed/`: Cleaned and featurized datasets (e.g. normalized descriptors, train/test splits, serialized feature matrices). Git-ignored except `.gitkeep`.

## Recommended Datasets for Hackathon Prototype
1. **PDBbind Core Set / Refined Set**: High-quality 3D structures and experimentally determined binding affinities ($pK_d / pK_i$).
2. **BindingDB Subsets**: 2D ligand SMILES with target protein sequences and binding constants ($IC_{50}, K_i$).
3. **Synthetic / Curated Benchmark**: Small subset (~100-500 samples) ideal for quick prototyping and quantum simulation constraints (qubit count and circuit depth limits).
