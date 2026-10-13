import numpy as np
import pytest
from io import StringIO
from Bio.PDB import PDBParser

from backbone_qc.metrics import calculate_radius_of_gyration


def test_radius_of_gyration_for_known_coordinates():
    coordinates = [
        [0, 0, 0],
        [2, 0, 0],
    ]

    result = calculate_radius_of_gyration(coordinates)

    assert result == pytest.approx(1.0)


def test_radius_of_gyration_is_independent_of_translation():
    coordinates = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
    ])

    shifted = coordinates + np.array([10, -5, 3])

    assert calculate_radius_of_gyration(coordinates) == pytest.approx(
        calculate_radius_of_gyration(shifted)
    )


def test_invalid_coordinates_raise_error():
    with pytest.raises(ValueError):
        calculate_radius_of_gyration([[0, 0], [1, 1]])


def test_single_coordinate_raises_error():
    with pytest.raises(ValueError):
        calculate_radius_of_gyration([[0, 0, 0]])

from backbone_qc.metrics import check_backbone_continuity


def test_normal_ca_spacing_is_continuous():
    coordinates = [
        [0, 0, 0],
        [3.8, 0, 0],
        [7.6, 0, 0],
    ]

    result = check_backbone_continuity(coordinates)

    assert result["n_flagged"] == 0


def test_large_ca_gap_is_flagged():
    coordinates = [
        [0, 0, 0],
        [3.8, 0, 0],
        [10.0, 0, 0],
    ]

    result = check_backbone_continuity(coordinates)

    assert result["n_flagged"] == 1
    assert result["flagged_indices"] == [1]


def test_single_ca_has_no_bonds():
    result = check_backbone_continuity([[0, 0, 0]])

    assert result["distances"] == []
    assert result["n_flagged"] == 0

    from Bio.PDB import PDBParser
from io import StringIO

from backbone_qc.metrics import calculate_backbone_angles


def test_calculate_backbone_angles_returns_phi_psi():
    pdb_text = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       1.950   1.400   0.000  1.00 20.00           C
ATOM      4  N   GLY A   2       1.300   2.400   0.000  1.00 20.00           N
ATOM      5  CA  GLY A   2       1.800   3.700   0.500  1.00 20.00           C
ATOM      6  C   GLY A   2       3.200   3.800   0.800  1.00 20.00           C
ATOM      7  N   ALA A   3       3.800   2.700   1.000  1.00 20.00           N
ATOM      8  CA  ALA A   3       5.100   2.800   1.500  1.00 20.00           C
ATOM      9  C   ALA A   3       5.600   4.100   2.000  1.00 20.00           C
TER
END
"""
    structure = PDBParser(QUIET=True).get_structure(
        "test", StringIO(pdb_text)
    )
    residues = list(structure[0]["A"])

    angles = calculate_backbone_angles(residues)

    assert len(angles) == 3
    assert angles[0]["phi_deg"] is None
    assert angles[0]["psi_deg"] is not None
    assert angles[1]["phi_deg"] is not None
    assert angles[1]["psi_deg"] is not None
    assert angles[2]["phi_deg"] is not None
    assert angles[2]["psi_deg"] is None

def test_backbone_angles_skip_peptide_chain_break():
    pdb_text = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       1.950   1.400   0.000  1.00 20.00           C
ATOM      4  N   GLY A   2      10.000  10.000  10.000  1.00 20.00           N
ATOM      5  CA  GLY A   2      11.450  10.000  10.000  1.00 20.00           C
ATOM      6  C   GLY A   2      11.950  11.400  10.000  1.00 20.00           C
TER
END
"""
    structure = PDBParser(QUIET=True).get_structure(
        "break_test", StringIO(pdb_text)
    )
    residues = list(structure[0]["A"])

    angles = calculate_backbone_angles(residues)

    assert len(angles) == 2
    assert angles[0]["psi_deg"] is None
    assert angles[1]["phi_deg"] is None