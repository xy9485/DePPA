#!/usr/bin/env python3
"""Fill PoseCheck strain/clash metrics into raw_scores.csv for a single pocket."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from openbabel import pybel

from analysis.eval_rmsd import get_rmsd_between_mol_pdbqt, get_rmsd_between_mols, get_mol_from_pdbqt_file, pdbqt_to_rdmol_openbabel


DESIRED_OUTPUT_COLUMNS = [
    "rank",
    "num_atoms",
    "qed",
    "sa",
    "distance",
    "distance_post",
    "strain",
    "clash",
    "connectivity",
    "vina_score",
    "vina_min",
    "vina_dock",
    "sc_rmsd",
]

# PoseCheck lives outside this repository, so fail early with a clear error if it is missing.
try:
    from posecheck import PoseCheck
    from posecheck.utils import loading as posecheck_loading
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "PoseCheck is required for this script. Please install posecheck before running."
    ) from exc

import time
import os
if os.getenv('ENABLE_DEBUG', 'false').lower() == 'true':
    import debugpy

    # Use any open port, e.g., 5678
    debugpy.listen(("0.0.0.0", 5675))
    print("🔍 Waiting for debugger attach on port 5675...")
    debugpy.wait_for_client()

def _compute_diversity_for_mol_list(mol_list):
    fps = [Chem.RDKFingerprint(mol) for mol in mol_list]
    distances = []
    for i, fp in enumerate(fps):
        sims = [
            DataStructs.TanimotoSimilarity(fp, other_fp)
            for j, other_fp in enumerate(fps)
            if j != i
        ]
        distances.append(0.0 if not sims else 1.0 - float(np.mean(sims)))
    return distances

def _load_molecules(sdf_path: Path) -> List[Optional[Chem.Mol]]:
    """Read molecules from an SDF file while keeping their original order."""
    if not sdf_path.exists():
        raise FileNotFoundError(f"Missing SDF file: {sdf_path}")

    # supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    # supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False) # earlier version uses sanitize=False when writing to sdf
    supplier = Chem.SDMolSupplier(str(sdf_path)) # arguments used for reading from sdf should align with how it was written
    # supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    # supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    molecules: List[Optional[Chem.Mol]] = []
    for idx, mol in enumerate(supplier):
        if mol is None:
            print(f"[WARN] Failed to parse molecule #{idx + 1} in {sdf_path.name}.")
            raise ValueError(f"Failed to parse molecule #{idx + 1} in {sdf_path.name}.")
        # else:
        #     try:
        #         Chem.SanitizeMol(mol)
        #     except Exception as exc:
        #         print(
        #             f"[WARN] Sanitization failed for molecule #{idx + 1} in {sdf_path.name}: {exc}"
        #         )
        #         raise ValueError(f"Sanitization failed for molecule #{idx + 1} in {sdf_path.name}: {exc}")
        molecules.append(mol)

    if not molecules:
        print(f"[WARN] No molecules were found in {sdf_path}.")

    return molecules

def _ligand_centroid_backup(mol: Chem.Mol) -> Optional[Tuple[float, float, float]]:
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    count = mol.GetNumAtoms()
    if count == 0:
        return None
    x = y = z = 0.0
    for atom_idx in range(count):
        pos = conf.GetAtomPosition(atom_idx)
        x += pos.x
        y += pos.y
        z += pos.z
    return (x / count, y / count, z / count)

def _ligand_centroid(mol: Chem.Mol) -> Optional[Tuple[float, float, float]]:
    try:
        coords = np.array(mol.GetConformer().GetPositions(), dtype=float)
        centroid = coords.mean(axis=0)
        return (float(centroid[0]), float(centroid[1]), float(centroid[2]))
    except Exception as exc:
        print(f"Failed to read coordinates from reference ligand")


def _locate_reference_ligand_sdf(receptor_pdbqt: Path) -> Optional[Path]:
    """Find an SDF file containing the reference ligand for the receptor."""
    directory = receptor_pdbqt.parent
    stem = receptor_pdbqt.stem

    candidates = sorted(directory.glob(f"{stem}*.sdf"))
    if not candidates:
        candidates = sorted(directory.glob(f"*{stem}*.sdf"))

    if not candidates:
        print(
            f"[WARN] Unable to find reference ligand SDF alongside {receptor_pdbqt.name}."
        )
        return None

    if len(candidates) > 1:
        print(
            f"[WARN] Multiple reference SDF files found for {stem}; using {candidates[0].name}."
        )
    return candidates[0]


def _load_reference_centroid(receptor_pdbqt: Path) -> Optional[Tuple[float, float, float]]:
    sdf_path = _locate_reference_ligand_sdf(receptor_pdbqt)
    if not sdf_path:
        return None

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    for idx, mol in enumerate(supplier):
        if mol is None:
            continue
        # try:
        #     Chem.SanitizeMol(mol)
        # except Exception as exc:
        #     print(
        #         f"[WARN] Sanitization failed for reference ligand ({sdf_path.name}, entry {idx + 1}): {exc}"
        #     )
        #     continue
        centroid = _ligand_centroid(mol)
        if centroid is not None:
            print(
                f"[INFO] Using fixed centroid from reference ligand {sdf_path.name} for vina."
            )
            return centroid

    print(
        f"[WARN] Unable to compute centroid from reference ligand SDF ({sdf_path.name})."
    )
    return None


def _parse_qvina_score(output: str) -> float:
    marker = "-----+------------+----------+----------"
    lines = output.splitlines()
    if marker in lines:
        try:
            best_idx = lines.index(marker) + 1
            best_line = lines[best_idx].split()
            if best_line and best_line[0] == "1":
                return float(best_line[1])
        except Exception:
            raise ValueError("Unable to parse qvina2 score from output table.")

    match = re.search(r"Affinity:\s*([-+]?\d*\.?\d+)", output)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            raise ValueError("Unable to parse qvina2 score from output affinity line.")
    return math.nan


def _run_qvina(
    mol: Chem.Mol,
    receptor_pdbqt: Path,
    centroid: Tuple[float, float, float],
    size: float,
    exhaustiveness: int,
    vina_mode: str = "vina_dock",
    compute_rmsd: bool = False,
    use_meeko: bool = False,
) -> Tuple[float, float]:
    assert vina_mode in {"vina_dock", "vina_min", "vina_score"}
    cx, cy, cz = centroid
    score = math.nan
    symmetry_rmsd = math.nan
    
    with tempfile.TemporaryDirectory(prefix="qvina_temp_") as tmpdir:
        lig_pdbqt = Path(tmpdir) / "ligand.pdbqt"
        out_pdbqt = Path(tmpdir) / "ligand_out.pdbqt"

        if use_meeko:
            from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule

            # Prepare the ligand using Meeko
            mk_prep = MoleculePreparation(merge_these_atom_types=("H",))
            mol = Chem.AddHs(mol, addCoords=mol.GetNumConformers() > 0)
            molsetup_list = mk_prep(mol)
            mol_setup = molsetup_list[0]
            pdbqt_string, success, error_msg = PDBQTWriterLegacy.write_string(mol_setup)

            # Write the prepared ligand to a PDBQT file
            with open(lig_pdbqt, "w") as f:
                f.write(pdbqt_string)
        else:
            molblock = Chem.MolToMolBlock(mol, kekulize=False)
            obmol = pybel.readstring("sdf", molblock)
            obmol.write("pdbqt", str(lig_pdbqt), overwrite=True)

        cmd = [
            "qvina2",
            "--receptor",
            str(receptor_pdbqt),
            "--ligand",
            str(lig_pdbqt),
            "--center_x",
            f"{cx:.4f}",
            "--center_y",
            f"{cy:.4f}",
            "--center_z",
            f"{cz:.4f}",
            "--size_x",
            str(size),
            "--size_y",
            str(size),
            "--size_z",
            str(size),
            "--exhaustiveness",
            str(exhaustiveness),
            "--out",
            str(out_pdbqt),
        ]
        if vina_mode == "vina_dock":
            pass
        if vina_mode == "vina_min":
            cmd.append("--local_only")
        if vina_mode == "vina_score":
            cmd.append("--score_only")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"[WARN] qvina2 failed (code {result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
            )
            return score, symmetry_rmsd

        score = _parse_qvina_score(result.stdout)
        if math.isnan(score):
            # print("[WARN] Unable to parse qvina2 score from output.")
            raise ValueError("Unable to parse qvina2 score from output, although qvina2 returns successfully.")

        if vina_mode == "vina_dock" and compute_rmsd:
            if not out_pdbqt.exists():
                raise FileNotFoundError("qvina2 did not produce an output PDBQT; skipping RMSD.")
            else:
                if use_meeko:
                    from meeko import PDBQTMolecule, RDKitMolCreate
                    # with open(out_pdbqt, 'r') as f:
                    #         pdbqt_data = f.read()
                    # pdbqt_mol = PDBQTMolecule(pdbqt_data)
                    # # Because the metadata exists, this mol will have the SAME atom order as mol before redocking,
                    # # Exporting directly to RDKit bypasses the 'lossy' PDB/SDF conversion steps
                    # # We take the 0th index for the best pose
                    # docked_mol = pdbqt_mol[0].export_rdkit_mol()
                    pdbqt_mol = PDBQTMolecule.from_file(str(out_pdbqt), skip_typing=True)
                    rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
                    docked_mol = rdkitmol_list[0]
                else:
                    # symmetry_rmsd = get_rmsd_between_mol_pdbqt(mol, str(out_pdbqt))
                    docked_mol = pdbqt_to_rdmol_openbabel(str(out_pdbqt))
                    # docked_mol = get_mol_from_pdbqt_file(str(out_pdbqt))
                    if docked_mol is None:
                        raise ValueError("Failed to convert qvina output PDBQT to RDMol.")
                # docked_mol_aligned = Chem.AllChem.AssignBondOrdersFromTemplate(mol, docked_mol)
                symmetry_rmsd = get_rmsd_between_mols(mol, docked_mol)

    return score, symmetry_rmsd


def _calculate_vina_metrics_for_molecules(
    molecules: List[Optional[Chem.Mol]],
    receptor_pdbqt: Path,
    size: float,
    exhaustiveness: int,
    fixed_centroid: Optional[Tuple[float, float, float]] = None,
) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []
    for idx, mol in enumerate(molecules):
        print(f"Vina Metrics, Processing molecule #{idx + 1} with size {mol.GetNumAtoms()} atoms.")
        entry = {
            "vina_dock": math.nan,
            "sc_rmsd": math.nan,
            "vina_min": math.nan,
            "vina_score": math.nan,
        }
        if mol is None:
            raise ValueError(f"Molecule #{idx + 1} is None; cannot compute vina metrics.")
            # metrics.append(entry)
            # continue

        centroid = fixed_centroid or _ligand_centroid(mol)
        if centroid is None:
            raise ValueError(f"Unable to compute centroid for molecule #{idx + 1}.")
            # metrics.append(entry)
            # continue


        vina_dock_score, rmsd = _run_qvina(
            mol=mol,
            receptor_pdbqt=receptor_pdbqt,
            centroid=centroid,
            size=size,
            exhaustiveness=exhaustiveness,
            vina_mode="vina_dock",
            compute_rmsd=True,
            use_meeko=False,
        )
        if not math.isnan(vina_dock_score):
            entry["vina_dock"] = vina_dock_score
        if not math.isnan(rmsd):
            entry["sc_rmsd"] = rmsd

        vina_min_score, _ = _run_qvina(
            mol=mol,
            receptor_pdbqt=receptor_pdbqt,
            centroid=centroid,
            size=size,
            exhaustiveness=exhaustiveness,
            vina_mode="vina_min",
            compute_rmsd=False,
        )
        if not math.isnan(vina_min_score):
            entry["vina_min"] = vina_min_score

        vina_score, _ = _run_qvina(
            mol=mol,
            receptor_pdbqt=receptor_pdbqt,
            centroid=centroid,
            size=size,
            exhaustiveness=exhaustiveness,
            vina_mode="vina_score",
            compute_rmsd=False,
        )
        if not math.isnan(vina_score):
            entry["vina_score"] = vina_score
        
        print(f"sc_rmsd: {entry['sc_rmsd']}, vina_dock: {entry['vina_dock']}, vina_min: {entry['vina_min']}, vina_score: {entry['vina_score']}")

        metrics.append(entry)

    return metrics


def _ensure_metric_column_order(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict output CSV to the desired schema and column order."""
    return df.reindex(columns=DESIRED_OUTPUT_COLUMNS)


def _run_posecheck2(
    molecules: List[Optional[Chem.Mol]], protein
) -> List[Dict[str, float]]:
    """Compute strain/clash metrics for every molecule using PoseCheck."""
    pc = PoseCheck()
    try:
        pc.protein = protein
    except ValueError as exc:
        print(f"[ERROR] Unable to load PoseCheck protein ({exc}).")
        raise ValueError(f"Unable to load PoseCheck protein ({exc}).") from exc
        # return [{"strain": math.nan, "clash": math.nan} for _ in molecules]

    results: List[Dict[str, float]] = []
    for idx, mol in enumerate(molecules):
        print(f"Posecheck processing molecule #{idx + 1} with size {mol.GetNumAtoms()} atoms.")
        entry = {"strain": math.nan, "clash": math.nan}
        if mol is None:
            raise ValueError(f"Molecule #{idx + 1} is None; cannot compute PoseCheck metrics.")

        try:
            pc.load_ligands_from_mols([mol])
            print("Ligand loaded into PoseCheck.")
        except ValueError as exc:
            print(
                f"[WARN] Failed to load ligand #{idx + 1} into PoseCheck (skipping): {exc}"
            )
            print(f"Strain: {entry['strain']}, Clash: {entry['clash']}")
            results.append(entry)
            continue

        try:
            entry["strain"] = pc.calculate_strain_energy()[0]
        except Exception as exc:
            print(f"[WARN] PoseCheck strain failed on entry #{idx + 1}: {exc}")

        try:
            entry["clash"] = pc.calculate_clashes()[0]
        except Exception as exc:
            print(f"[WARN] PoseCheck clash failed on entry #{idx + 1}: {exc}")

        print(f"Strain: {entry['strain']}, Clash: {entry['clash']}")
        results.append(entry)

    return results

def _run_posecheck(
    molecules: List[Optional[Chem.Mol]], protein
) -> List[Dict[str, float]]:
    """Compute strain/clash metrics for every molecule using PoseCheck."""
    pc = PoseCheck()
    pc.protein = protein

    results = {}
    pc.load_ligands_from_mols(molecules)
    results['strain'] = pc.calculate_strain_energy()
    results['clash'] = pc.calculate_clashes()
    return results



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
        "--dir_postfill",
        type=Path,
        help="search for raw.sdf and raw_scores.csv under this folder",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default=None,
        help="Filename of the csv file to process",
    )
    parser.add_argument(
        "--sdf_name",
        type=str,
        default=None,
        help="Filename of the SDF file to process",
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
        "--pocket_pdbqt_dir",
        type=Path,
        help="Directory containing receptor .pdbqt files (defaults to --pocket_pdb_dir).",
    )
    parser.add_argument(
        "--out_csv_name",
        type=str,
        default=None,
        help="Filename to write alongside raw_scores.csv with filled PoseCheck metrics.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing --output-name file.",
    )
    # parser.add_argument(
    #     "--qvina_size",
    #     type=float,
    #     default=20.0,
    #     help="Box size (Å) for qvina2 runs used to compute symmetry RMSD (default: 20).",
    # )
    # parser.add_argument(
    #     "--qvina_exhaustiveness",
    #     type=int,
    #     default=16,
    #     help="qvina2 exhaustiveness level for RMSD docking (default: 16).",
    # )
    parser.set_defaults(compute_posecheck=True, compute_vina=True, compute_diversity=True)
    parser.add_argument(
        "--compute_posecheck",
        dest="compute_posecheck",
        action="store_true",
        help="Compute PoseCheck strain/clash metrics (default: enabled).",
    )
    parser.add_argument(
        "--skip_posecheck",
        dest="compute_posecheck",
        action="store_false",
        help="Disable PoseCheck strain/clash metrics.",
    )
    parser.add_argument(
        "--compute_vina",
        dest="compute_vina",
        action="store_true",
        help="Compute qvina2-derived metrics (default: enabled).",
    )
    parser.add_argument(
        "--skip_vina",
        dest="compute_vina",
        action="store_false",
        help="Disable qvina2-derived metrics.",
    )
    parser.add_argument(
        "--compute_diversity",
        dest="compute_diversity",
        action="store_true",
        help="Compute molecular diversity metrics (default: enabled).",
    )
    parser.add_argument(
        "--skip_diversity",
        dest="compute_diversity",
        action="store_false",
        help="Disable molecular diversity metrics.",
    )
    parser.set_defaults(vina_use_reflig_centroid=True)
    parser.add_argument(
        "--vina_use_reflig_centroid",
        dest="vina_use_reflig_centroid",
        action="store_true",
        help=(
            "Use the centroid of the receptor's reference ligand SDF (stored next to the PDBQT) "
            "for all docking runs instead of per-molecule centroids. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no_vina_use_reflig_centroid",
        dest="vina_use_reflig_centroid",
        action="store_false",
        help="Disable using the reference ligand centroid for docking calculations.",
    )
    return parser.parse_args()


def post_fill_metrics(dir_post_fill: Path, csv_name: str, sdf_name: str=None, pocket_pdb_dir: Path=None, compute_vina: bool=True, vina_use_reflig_centroid: bool=True, compute_posecheck: bool=True, compute_diversity: bool=True, out_csv_name: str=None, overwrite: bool=False) -> None:

    csv_path: Path = dir_post_fill / csv_name
    # assert csv_path.name.endswith("_scores.csv")
    if sdf_name is None:
        # sdf_name share the same name with csv_name but with .sdf extension
        sdf_name = csv_path.stem + ".sdf"
    sdf_path = dir_post_fill / sdf_name
    assert sdf_path.exists(), f"SDF file does not exist: {sdf_path}"
    assert csv_path.exists(), f"CSV file does not exist: {csv_path}"
    pocket_name = dir_post_fill.name
    assert pocket_name.endswith("pocket10")

    print(f"[INFO] Using ligand SDF: {sdf_path}")
    print(f"[INFO] Using raw scores CSV: {csv_path}")
    print(f"[INFO] Pocket name: {pocket_name}")

    molecules = _load_molecules(sdf_path)
    print(f"[INFO] Loaded {len(molecules)} molecules from SDF.")
    raw_scores = pd.read_csv(csv_path)
    # raise error if molecules contains None
    if any(mol is None for mol in molecules):
        raise ValueError("One or more molecules failed to load from the SDF file.")
    if len(raw_scores) != len(molecules):
        raise ValueError(
            f"Molecule count ({len(molecules)}) differs from CSV rows ({len(raw_scores)})."
        )

    pocket_pdbqt_dir: Path = pocket_pdb_dir

    if compute_vina:
        receptor_path_pdbqt = pocket_pdbqt_dir / f"{pocket_name}.pdbqt"
        print(f"[Vina] Using receptor PDBQT: {receptor_path_pdbqt}")
        # receptor_path_pdbqt = _locate_receptor_pdbqt(pocket_name, pocket_pdbqt_dir)
        reference_centroid: tuple[float, float, float] | None = None
        if vina_use_reflig_centroid:
            ref_lig_sdf = next(receptor_path_pdbqt.parent.glob(f"{receptor_path_pdbqt.stem}*.sdf"))
            # reference_centroid = _load_reference_centroid(receptor_path_pdbqt)
            reference_centroid = _ligand_centroid(Chem.SDMolSupplier(str(ref_lig_sdf))[0])
        
        print("[Vina] Start Running qvina2 metrics...")
        vina_metrics = _calculate_vina_metrics_for_molecules(
            molecules=molecules,
            receptor_pdbqt=receptor_path_pdbqt,
            size=20.0,
            exhaustiveness=16,
            fixed_centroid=reference_centroid,
        )
        raw_scores["vina_dock"] = [
            entry.get("vina_dock", math.nan) for entry in vina_metrics
        ]
        raw_scores["sc_rmsd"] = [
            entry.get("sc_rmsd", math.nan) for entry in vina_metrics
        ]
        raw_scores["vina_min"] = [
            entry.get("vina_min", math.nan) for entry in vina_metrics
        ]
        raw_scores["vina_score"] = [
            entry.get("vina_score", math.nan) for entry in vina_metrics
        ]

    if compute_posecheck:
        # receptor_path_pdb = _locate_receptor_pdb(pocket_name, pocket_pdb_dir)
        receptor_path_pdb = pocket_pdb_dir / f"{pocket_name}.pdb"
        protein = posecheck_loading.load_protein_from_pdb(str(receptor_path_pdb))
        print(f"[Posecheck] Using receptor pdb: {receptor_path_pdb}")
        print("[Posecheck] Start Running PoseCheck...")
        posecheck_results = _run_posecheck2(molecules, protein)
        assert len(posecheck_results) == len(molecules)
        raw_scores["strain"] = [
            entry.get("strain", math.nan) for entry in posecheck_results
        ]
        raw_scores["clash"] = [
            entry.get("clash", math.nan) for entry in posecheck_results
        ]
        # posecheck_results = _run_posecheck(molecules, protein)
        # assert len(posecheck_results['strain']) == len(molecules)
        # raw_scores["strain"] = posecheck_results['strain']
        # raw_scores["clash"] = posecheck_results['clash']

    if compute_diversity:
        print("[Diversity] Start Computing diversity metrics...")
        distances = _compute_diversity_for_mol_list(molecules)
        assert len(distances) == len(molecules)
        raw_scores["distance_post"] = distances
    if out_csv_name:
        output_path = csv_path.with_name(out_csv_name)
    else:
        #output_path is csv_path with "_postfilled" appended before the .csv extension
        output_path = csv_path.with_name(csv_path.stem + "_postfilled.csv")
    if output_path.exists() and not overwrite:
        print(f"[INFO] Updating existing metrics file at {output_path}.")
        existing_scores = pd.read_csv(output_path)
        if len(existing_scores) != len(raw_scores):
            raise ValueError(
                f"Existing CSV rows ({len(existing_scores)}) do not match current rows ({len(raw_scores)})."
            )
        if compute_posecheck:
            existing_scores["strain"] = raw_scores["strain"]
            existing_scores["clash"] = raw_scores["clash"]
        if compute_vina:
            existing_scores["vina_score"] = raw_scores["vina_score"]
            existing_scores["vina_min"] = raw_scores["vina_min"]
            existing_scores["vina_dock"] = raw_scores["vina_dock"]
            existing_scores["sc_rmsd"] = raw_scores["sc_rmsd"]
        existing_scores = _ensure_metric_column_order(existing_scores)
        existing_scores.to_csv(output_path, index=False)
    else:
        raw_scores = _ensure_metric_column_order(raw_scores)
        raw_scores.to_csv(output_path, index=False)
    print(f"[DONE] Wrote PoseCheck metrics to {output_path}")


if __name__ == "__main__":
    args = _parse_args()
    # --results_dir or --dir_postfill must be specified. only one of them is not none at a time.
    assert (
        (args.results_dir is None) ^ (args.dir_postfill is None)
    ), "Exactly one of --results_dir or --dir_postfill must be specified"

    start_time = time.time()
    if args.dir_postfill:
        post_fill_metrics(
            dir_post_fill=args.dir_postfill,
            csv_name=args.csv_name,
            sdf_name=args.sdf_name,
            pocket_pdb_dir=args.pocket_pdb_dir,
            compute_vina=args.compute_vina,
            vina_use_reflig_centroid=args.vina_use_reflig_centroid,
            compute_posecheck=args.compute_posecheck,
            compute_diversity=args.compute_diversity,
            out_csv_name=args.out_csv_name,
            overwrite=args.overwrite,
        )
    else:

        for idx, pocket_dir in enumerate(sorted(args.results_dir.iterdir())):
            if not pocket_dir.is_dir():
                continue
            print(f"[INFO] ========Processing {idx}th pocket directory: {pocket_dir.name}========")
            post_fill_metrics(
                dir_post_fill=pocket_dir,
                csv_name=args.csv_name,
                sdf_name=args.sdf_name,
                pocket_pdb_dir=args.pocket_pdb_dir,
                compute_vina=args.compute_vina,
                vina_use_reflig_centroid=args.vina_use_reflig_centroid,
                compute_posecheck=args.compute_posecheck,
                compute_diversity=args.compute_diversity,
                out_csv_name=args.out_csv_name,
                overwrite=args.overwrite,
            )
    end_time = time.time()
    print(f"[INFO] Total execution time: {end_time - start_time:.2f} seconds.")
