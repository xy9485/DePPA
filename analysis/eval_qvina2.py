#!/usr/bin/env python

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "qvina2_temp_output"
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)


def get_center_from_pdbqt(path: Path):
    xs, ys, zs = [], [], []
    with path.open() as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
    if not xs:
        raise ValueError(f"No atom coordinates found in {path}")
    n = len(xs)
    return sum(xs) / n, sum(ys) / n, sum(zs) / n


def parse_score(output: str) -> float:
    """Extract either the docking table score or 'Affinity' score from output."""
    lines = output.splitlines()
    separator = "-----+------------+----------+----------"
    if separator in lines:
        idx = lines.index(separator) + 1
        parts = lines[idx].split()
        return float(parts[1])

    for line in lines:
        if line.startswith("Affinity:"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1])
            break

    raise RuntimeError("Could not find an affinity score in QVina output.\n" + output)


def ensure_pdbqt(ligand: Path, workspace: Optional[Path] = None) -> Path:
    ligand = ligand.resolve()
    if ligand.suffix.lower() == ".pdbqt":
        return ligand
    if ligand.suffix.lower() == ".sdf":
        target_dir = workspace if workspace is not None else ligand.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        pdbqt_path = target_dir / f"{ligand.stem}.pdbqt"
        if pdbqt_path.exists():
            pdbqt_path.unlink()
        subprocess.run(
            ["obabel", str(ligand), "-O", str(pdbqt_path)],
            check=True,
        )
        return pdbqt_path
    raise ValueError(f"Unsupported ligand format: {ligand.suffix}")


def discover_ligand_pocket_pairs(directory: Path) -> List[tuple[Path, Path]]:
    """Return (receptor, ligand) pairs inferred from a directory of files."""
    directory = directory.resolve()
    pairs: List[tuple[Path, Path]] = []
    seen_ligands = set()

    def maybe_add_pair(ligand_path: Path):
        if "_" not in ligand_path.stem:
            return
        prefix = ligand_path.stem.split("_", 1)[0]
        receptor_path = directory / f"{prefix}.pdbqt"
        if not receptor_path.exists() or ligand_path == receptor_path:
            return
        resolved = ligand_path.resolve()
        if resolved in seen_ligands:
            return
        seen_ligands.add(resolved)
        pairs.append((receptor_path, ligand_path))

    for ligand_path in sorted(directory.glob("*_*.sdf")):
        maybe_add_pair(ligand_path)
    return pairs


def run_qvina(
    receptor: Path,
    ligand: Path,
    size: float,
    exhaustiveness: int,
    center=None,
    mode: str = "dock",
    output_path: Path = DEFAULT_OUTPUT_DIR,
) -> float:
    output_dir = output_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ligand = ensure_pdbqt(ligand, output_dir)

    if center is None:
        cx, cy, cz = get_center_from_pdbqt(ligand)
    else:
        cx, cy, cz = center

    cmd = [
        "qvina2",
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--center_x", f"{cx:.4f}",
        "--center_y", f"{cy:.4f}",
        "--center_z", f"{cz:.4f}",
        "--size_x", str(size),
        "--size_y", str(size),
        "--size_z", str(size),
        "--exhaustiveness", str(exhaustiveness),
    ]
    out_file = output_dir / f"{ligand.stem}_out.pdbqt"
    if out_file.exists():
        out_file.unlink()
    cmd.extend(["--out", str(out_file)])

    if mode == "score_only":
        cmd.append("--score_only")
    elif mode == "local_only":
        cmd.append("--local_only")
    elif mode != "dock":
        raise ValueError(f"Unsupported QVina mode: {mode}")

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    log_path = output_dir / f"qvina2_stdout_{mode}.txt"
    log_path.write_text(result.stdout)
    print(f"QVina output files saved in: {output_dir}")
    print(result.stdout)
    return parse_score(result.stdout)


def main():
    parser = argparse.ArgumentParser("QVina2 scoring from receptor + ligand")
    parser.add_argument("--receptor", type=Path, help="Receptor PDBQT file")
    parser.add_argument(
        "--ligand",
        type=Path,
        help="Ligand file (.pdbqt or .sdf)",
    )
    parser.add_argument("--size", type=float, default=20.0, help="Cubic box edge length (Å)")
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        metavar=("CX", "CY", "CZ"),
        help="Optional box center; default = ligand centroid",
    )
    parser.add_argument(
        "--mode",
        choices=["dock", "score_only", "local_only"],
        default="dock",
        help="QVina mode: dock (default), score-only, or local-only minimization",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where QVina output files are stored (default: qvina2_temp_output)",
    )
    parser.add_argument(
        "--ligand_pocket_directory",
        type=Path,
        help="Directory containing receptor (.pdbqt) and ligand (.sdf/.pdbqt) files; "
        "all detected pairs will be processed when provided.",
    )
    args = parser.parse_args()

    label = {
        "dock": "Best QVina2 score",
        "score_only": "QVina2 score-only affinity",
        "local_only": "QVina2 local-only affinity",
    }[args.mode]

    if args.ligand_pocket_directory:
        directory = args.ligand_pocket_directory
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} is not a directory")
        pairs = discover_ligand_pocket_pairs(directory)
        if not pairs:
            raise ValueError(f"No receptor/ligand pairs found in {directory}")
        for receptor_path, ligand_path in pairs:
            pair_output = args.output_path / receptor_path.stem
            print(f"Processing receptor '{receptor_path.name}' with ligand '{ligand_path.name}'")
            score = run_qvina(
                receptor_path,
                ligand_path,
                args.size,
                args.exhaustiveness,
                args.center,
                args.mode,
                pair_output,
            )
            print(f"{label} for {ligand_path.name}: {score:.3f} kcal/mol")
        return

    if args.receptor is None or args.ligand is None:
        parser.error("Specify both --receptor and --ligand unless --ligand_pocket_directory is used.")

    score = run_qvina(
        args.receptor,
        args.ligand,
        args.size,
        args.exhaustiveness,
        args.center,
        args.mode,
        args.output_path,
    )
    print(f"{label}: {score:.3f} kcal/mol")


if __name__ == "__main__":
    main()
