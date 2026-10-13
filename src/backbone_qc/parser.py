from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser


def parse_structure(pdb_path):
    """Read a PDB file and extract backbone data for each chain."""
    pdb_path = Path(pdb_path)

    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    models = list(structure.get_models())
    if not models:
        raise ValueError("No structural model found in PDB file.")

    model = models[0]
    chains = []

    for chain in model:
        residues = []
        ca_coordinates = []

        for residue in chain:
            # Keep standard amino-acid residues only.
            if residue.id[0] != " ":
                continue

            # C-alpha is required for the existing metrics.
            if "CA" not in residue:
                continue

            ca_coord = np.asarray(residue["CA"].coord, dtype=float)
            if not np.isfinite(ca_coord).all():
                continue

            residues.append(residue)
            ca_coordinates.append(ca_coord)

        if residues:
            chains.append({
                "chain_id": chain.id,
                "residues": residues,
                "ca_coordinates": np.asarray(ca_coordinates, dtype=float),
                "n_residues": len(residues),
            })

    if not chains:
        raise ValueError("No protein C-alpha atoms found in PDB file.")

    return {
        "file": pdb_path.name,
        "chains": chains,
        "n_chains": len(chains),
        "n_residues": sum(c["n_residues"] for c in chains),
    }
