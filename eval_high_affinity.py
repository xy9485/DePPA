import argparse
import csv
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem

from post_metrics import _calculate_vina_metrics_for_molecules, _ligand_centroid, _load_molecules


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill PoseCheck strain/clash values into raw_scores.csv for a single pocket."
        ),
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        help="Directory that contain subdirectorys that contain results for each pocket.",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default=None,
        help="Filename of the csv file to process",
    )
    parser.add_argument(
        "--pocket_pdb_dir",
        type=Path,
        default=Path("/home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test"),
        help=(
            "Directory containing receptor files (<pocket>.pdbqt or <pocket>.pdb) "
            "that match the parent folder name of the SDF."
        ),
    )
    parser.add_argument(
        "--ref_affinity_csv",
        type=Path,
        default=Path("affinity_ref.csv"),
        help="CSV file containing reference affinities for each pocket.",
    )
    return parser.parse_args()


def eval_ref_affinity(args, save: bool = True):
    args = _parse_args()
    print(args)
    pocket_pdb_dir = args.pocket_pdb_dir
    sorted_files = sorted(pocket_pdb_dir.iterdir())
    # filter by suffix "pdbqt"
    sorted_pdbqt = [f for f in sorted_files if f.suffix == ".pdbqt"]
    assert len(sorted_pdbqt) == 100, "Expected 100 pdbqt files in the pocket_pdb_dir."
    ref_affinity_dict = {"pocket_name": [], "vina_score": [], "vina_min": [], "vina_dock": [], "sc_rmsd": []}
    for idx, pocket_pdbqt in enumerate(sorted_pdbqt):
        print(f"Processing {idx+1}/100: {pocket_pdbqt.name}")
        pocket_name = pocket_pdbqt.stem
        ref_lig_sdf = next(pocket_pdb_dir.glob(f"{pocket_name}*.sdf"))
        ref_centroid = _ligand_centroid(Chem.SDMolSupplier(str(ref_lig_sdf))[0])
        ref_mol = Chem.SDMolSupplier(str(ref_lig_sdf))[0]
        
        vina_metrics = _calculate_vina_metrics_for_molecules(
            molecules=[ref_mol],
            receptor_pdbqt=pocket_pdbqt,
            size=20.0,
            exhaustiveness=8,
            fixed_centroid=False,
        )
        vina_score = vina_metrics[0]["vina_score"]
        vina_min = vina_metrics[0]["vina_min"]
        vina_dock = vina_metrics[0]["vina_dock"]
        sc_rmsd = vina_metrics[0]["sc_rmsd"]

        ref_affinity_dict["pocket_name"].append(pocket_name)
        ref_affinity_dict["vina_score"].append(vina_score)
        ref_affinity_dict["vina_min"].append(vina_min)
        ref_affinity_dict["vina_dock"].append(vina_dock)
        ref_affinity_dict["sc_rmsd"].append(sc_rmsd)
    
    # save to csv
    ref_affinity_df = pd.DataFrame(ref_affinity_dict)
    ref_affinity_df.to_csv("affinity_ref.csv", index=True)

def eval_high_affinity(args):
    ref_affinity_csv = args.ref_affinity_csv
    with ref_affinity_csv.open("r", newline="") as f:
        red_affinity_csv_reader = csv.DictReader(f)
        ref_affinity_rows = list(red_affinity_csv_reader)
    
    high_affinity_counts = []
    generated_mol_counts = []
    for row_idx, ref_row in enumerate(ref_affinity_rows):
        print(f"Processing pocket {row_idx+1}/100: {ref_row['pocket_name']}")
        sorted_dirs = sorted(args.results_dir.iterdir())
        num_dirs = sum(1 for d in sorted_dirs if d.is_dir())
        # assert num_dirs == 100, f"Expected 100 pocket directories under {args.results_dir}, found {num_dirs}."
        for dir_per_pocket in sorted_dirs:
            if not dir_per_pocket.is_dir():
                continue
            if dir_per_pocket.name != ref_row["pocket_name"]:
                continue
            # load raw_scores.csv
            raw_scores_csv_path = dir_per_pocket / args.csv_name
            with raw_scores_csv_path.open("r", newline="") as f:
                raw_scores_csv_reader = csv.DictReader(f)
                raw_scores_rows = list(raw_scores_csv_reader)
            # count how many ligands have vina_score better than ref vina_score
            # ref_vina_score = float(ref_row["vina_score"])
            ref_vina_score = float(ref_row["vina_score"])
            high_affinity_count = sum(1 for r in raw_scores_rows if abs(float(r["vina_score"])) > abs(ref_vina_score))
            high_affinity_counts.append(high_affinity_count)
            generated_mol_counts.append(len(raw_scores_rows))
            # temp check len(raw_scores_rows) == 100
            # assert len(raw_scores_rows) == 100, f"Expected 100 generated ligands for pocket {dir_per_pocket.name}, found {len(raw_scores_rows)}."

            print(f"Pocket {ref_row['pocket_name']}: {high_affinity_count} ligands have better vina_score than reference ({ref_vina_score}).")
            # print(f"high affinity rate per pocket: {high_affinity_count}/{len(raw_scores_rows)} = {(high_affinity_count/len(raw_scores_rows))*100.0:.2f}%")
            break
    
    # compute high affinity ratio
    high_affinity_ratios = [
        (high_affinity_counts[i] / generated_mol_counts[i]) * 100.0
        for i in range(len(high_affinity_counts))
    ]
    # compute mean and median
    mean_high_affinity_ratio = sum(high_affinity_ratios) / len(high_affinity_ratios)
    median_high_affinity_ratio = sorted(high_affinity_ratios)[len(high_affinity_ratios) // 2]
    print(f"Mean high affinity ratio: {mean_high_affinity_ratio:.2f}%")
    print(f"Median high affinity ratio: {median_high_affinity_ratio:.2f}%")



if __name__ == "__main__":
    args = _parse_args()
    assert args.results_dir is not None, "Please provide --results_dir argument."
    assert args.csv_name is not None, "Please provide --csv_name argument."
    # eval_ref_affinity(args)
    eval_high_affinity(args)
