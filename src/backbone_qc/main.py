from pathlib import Path

from backbone_qc.metrics import (
    calculate_radius_of_gyration,
    check_backbone_continuity,
    calculate_backbone_angles,
)
from backbone_qc.parser import parse_structure
from backbone_qc.report import save_results


INPUT_DIR = Path("input")
OUTPUT_DIR = Path("results")


def analyze_pdb(pdb_path):
    """Calculate initial quality metrics for each chain in a PDB file."""
    structure = parse_structure(pdb_path)
    results = []

    for chain in structure["chains"]:
        if chain["chain_id"] != "A":
            continue

        coords = chain["ca_coordinates"]
        angles = calculate_backbone_angles(chain["residues"])
        valid_phi = [item["phi_deg"] for item in angles if item["phi_deg"] is not None]
        valid_psi = [item["psi_deg"] for item in angles if item["psi_deg"] is not None]


        n_complete_angles = sum(
            item["phi_deg"] is not None and item["psi_deg"] is not None
            for item in angles
        )

        complete_angle_fraction = (
            n_complete_angles / len(angles) if angles else 0.0
        )
         
        rg = calculate_radius_of_gyration(coords)
        continuity = check_backbone_continuity(coords)

        results.append({
            "file": structure["file"],
            "chain_id": chain["chain_id"],
            "n_residues": chain["n_residues"],
            "radius_of_gyration_A": rg,
            "n_flagged_ca_distances": continuity["n_flagged"],
            "n_phi_angles": len(valid_phi),
            "n_psi_angles": len(valid_psi),
            "n_residues_with_phi_psi": n_complete_angles,
            "fraction_residues_with_phi_psi": complete_angle_fraction,
        })

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdb_files = sorted(INPUT_DIR.glob("*.pdb"))

    if not pdb_files:
        print(f"No PDB files found in {INPUT_DIR.resolve()}")
        return

    all_results = []

    for pdb_file in pdb_files:
        try:
            all_results.extend(analyze_pdb(pdb_file))
            print(f"Analyzed: {pdb_file.name}")
        except Exception as error:
            print(f"Failed: {pdb_file.name} — {error}")

    if all_results:
        output_path = OUTPUT_DIR / "backbone_metrics.csv"
        save_results(all_results, output_path)
        print(f"\nSaved results to: {output_path.resolve()}")
    else:
        print("No structures could be analyzed.")


if __name__ == "__main__":
    main()