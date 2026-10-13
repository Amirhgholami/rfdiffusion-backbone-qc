import pytest

from backbone_qc.parser import parse_structure


def test_parse_pdb_extracts_ca_coordinates(tmp_path):
    pdb_content = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00 20.00           C
ATOM      4  N   GLY A   2       3.000   1.500   0.000  1.00 20.00           N
ATOM      5  CA  GLY A   2       4.200   1.500   0.000  1.00 20.00           C
ATOM      6  C   GLY A   2       4.800   2.800   0.000  1.00 20.00           C
TER
END
"""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_content)

    result = parse_structure(pdb_file)

    assert result["n_chains"] == 1
    assert result["n_residues"] == 2
    assert result["chains"][0]["chain_id"] == "A"
    assert result["chains"][0]["ca_coordinates"].shape == (2, 3)


def test_missing_pdb_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_structure(tmp_path / "missing.pdb")


def test_pdb_without_ca_atoms_raises_error(tmp_path):
    pdb_file = tmp_path / "no_ca.pdb"
    pdb_file.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\nEND\n"
    )

    with pytest.raises(ValueError):
        parse_structure(pdb_file)