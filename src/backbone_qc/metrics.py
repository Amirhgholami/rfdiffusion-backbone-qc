import numpy as np


def calculate_radius_of_gyration(ca_coordinates):
    """
    Calculate the radius of gyration from C-alpha coordinates.

    Parameters
    ----------
    ca_coordinates : array-like, shape (n, 3)
        Cartesian coordinates of C-alpha atoms in Angstroms.

    Returns
    -------
    float
        Radius of gyration in Angstroms.
    """
    coordinates = np.asarray(ca_coordinates, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n, 3).")

    if len(coordinates) < 2:
        raise ValueError("At least two coordinates are required.")

    if not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must contain only finite values.")

    center = coordinates.mean(axis=0)
    squared_distances = np.sum((coordinates - center) ** 2, axis=1)

    return float(np.sqrt(squared_distances.mean()))

def check_backbone_continuity(ca_coordinates, min_distance=3.4, max_distance=4.2):
    """
    Flag unusually short or long distances between consecutive C-alpha atoms.

    Coordinates must be ordered by residue sequence within one continuous
    chain segment. Missing residues and chain breaks should be handled
    separately by the parser.

    Returns a dictionary containing distances and flagged bond indices.
    """
    coordinates = np.asarray(ca_coordinates, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n, 3).")

    if not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must contain only finite values.")

    if min_distance <= 0 or max_distance <= min_distance:
        raise ValueError("Distance thresholds are invalid.")

    if len(coordinates) < 2:
        return {
            "distances": [],
            "flagged_indices": [],
            "n_flagged": 0,
        }

    distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)

    flagged_indices = [
        i for i, distance in enumerate(distances)
        if distance < min_distance or distance > max_distance
    ]

    return {
        "distances": distances.tolist(),
        "flagged_indices": flagged_indices,
        "n_flagged": len(flagged_indices),
    }
def calculate_ca_geometry(ca_coordinates):
    """
    Calculate basic geometric descriptors from C-alpha coordinates.
    """
    coordinates = np.asarray(ca_coordinates, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n, 3).")

    if len(coordinates) < 2:
        raise ValueError("At least two coordinates are required.")

    if not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must contain only finite values.")

    pairwise_differences = (
        coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    )
    pairwise_distances = np.linalg.norm(pairwise_differences, axis=2)

    return {
        "max_ca_distance_A": float(pairwise_distances.max()),
        "mean_pairwise_ca_distance_A": float(
            pairwise_distances.mean()
        ),
    }

from Bio.PDB.vectors import calc_dihedral


def calculate_backbone_angles(residues):
    """
    Calculate phi and psi backbone angles for a continuous residue segment.

    Parameters
    ----------
    residues : list
        Ordered Biopython Residue objects from one continuous chain segment.

    Returns
    -------
    list of dict
        Residue number and phi/psi angles in degrees.
        Terminal or incomplete angles are returned as None.
    """
    results = []

    for i, residue in enumerate(residues):
        phi = None
        psi = None

        if i > 0:
            previous = residues[i - 1]

            if (
                "C" in previous
                and "N" in residue
                and "CA" in residue
                and "C" in residue
            ):
                phi = calc_dihedral(
                    previous["C"].get_vector(),
                    residue["N"].get_vector(),
                    residue["CA"].get_vector(),
                    residue["C"].get_vector(),
                )
                phi = float(np.degrees(phi))

        if i < len(residues) - 1:
            following = residues[i + 1]

            if (
                "N" in residue
                and "CA" in residue
                and "C" in residue
                and "N" in following
            ):
                psi = calc_dihedral(
                    residue["N"].get_vector(),
                    residue["CA"].get_vector(),
                    residue["C"].get_vector(),
                    following["N"].get_vector(),
                )
                psi = float(np.degrees(psi))

        results.append({
            "residue_number": residue.id[1],
            "phi_deg": phi,
            "psi_deg": psi,
        })

    return results

from Bio.PDB.vectors import calc_dihedral


def calculate_backbone_angles(residues, max_peptide_bond_distance=1.8):
    """Calculate phi/psi angles, avoiding obvious peptide-chain breaks."""
    results = []

    for i, residue in enumerate(residues):
        phi = None
        psi = None

        has_backbone = all(atom in residue for atom in ("N", "CA", "C"))

        if has_backbone and i > 0:
            previous = residues[i - 1]

            if all(atom in previous for atom in ("N", "CA", "C")):
                c_prev = previous["C"]
                n_curr = residue["N"]

                # Only calculate phi when the peptide bond is plausible.
                if c_prev - n_curr <= max_peptide_bond_distance:
                    phi = float(np.degrees(calc_dihedral(
                        c_prev.get_vector(),
                        n_curr.get_vector(),
                        residue["CA"].get_vector(),
                        residue["C"].get_vector(),
                    )))

        if has_backbone and i < len(residues) - 1:
            following = residues[i + 1]

            if all(atom in following for atom in ("N", "CA", "C")):
                c_curr = residue["C"]
                n_next = following["N"]

                # Only calculate psi when the peptide bond is plausible.
                if c_curr - n_next <= max_peptide_bond_distance:
                    psi = float(np.degrees(calc_dihedral(
                        residue["N"].get_vector(),
                        residue["CA"].get_vector(),
                        c_curr.get_vector(),
                        n_next.get_vector(),
                    )))

        results.append({
            "residue_number": residue.id[1],
            "phi_deg": phi,
            "psi_deg": psi,
        })

    return results