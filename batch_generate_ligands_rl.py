import argparse
import csv
import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import torch
from rdkit import Chem, DataStructs
from Bio.PDB import PDBParser
from openbabel import openbabel

import utils
from lightning_modules import LigandPocketDDPM

import os
if os.getenv('ENABLE_DEBUG', 'false').lower() == 'true':
    import debugpy

    # Use any open port, e.g., 5678
    debugpy.listen(("0.0.0.0", 5675))
    print("🔍 Waiting for debugger attach on port 5675...")
    debugpy.wait_for_client()

openbabel.obErrorLog.StopLogging()  # quiet OpenBabel output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RL-guided ligand generation for every pocket in a dataset and keep the top ligands."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to the pretrained diffusion checkpoint.")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("/home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test"),
        help="Directory containing pocket pdb/pdbqt, residue txt, and reference ligand sdf files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("rl_batch_outputs"),
        help="Where to store the top ligands and scores for each pocket.",
    )
    parser.add_argument("--n_samples", type=int, default=64, help="Samples per rollout.")
    parser.add_argument("--rollouts", type=int, default=10, help="Number of PPO rollouts per pocket.")
    parser.add_argument("--num_nodes_lig", type=int, default=None, help="Override ligand atom count.")
    parser.add_argument("--sanitize", action="store_true", help="Sanitize generated molecules.")
    parser.add_argument("--relax", action="store_true", help="Run force field relaxation.")
    parser.add_argument("--all_frags", action="store_true", help="Keep all fragments instead of only the largest one.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override diffusion timesteps.")
    parser.add_argument("--reward_workers", type=int, default=3, help="Workers for reward computation.")
    parser.add_argument("--ppo_lr", type=float, default=1e-5, help="PPO learning rate.")
    parser.add_argument("--ppo_batch_size", type=int, default=32, help="PPO minibatch size.")
    parser.add_argument("--inference_interval", type=int, default=10, help="Steps between PPO observations.")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="PPO grad clipping.")
    parser.add_argument("--kl_coeff_pretrain", type=float, default=0.0, help="KL penalty to the pretrained policy.")
    parser.add_argument("--top_k", type=int, default=100, help="How many ligands to retain per pocket.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of pockets to process.")
    return parser.parse_args()


def iter_pockets(dataset_dir: Path):
    for pdb_file in sorted(dataset_dir.glob("*.pdb")):
        base = pdb_file.stem
        receptor_file = pdb_file.with_suffix(".pdbqt")
        if not receptor_file.exists():
            raise FileNotFoundError(f"Receptor file not found: {receptor_file}")

        sdf_candidates = sorted(dataset_dir.glob(f"{base}*.sdf"))
        if not sdf_candidates:
            raise FileNotFoundError(f"No reference ligand sdf found for pocket {base}")
        ref_ligand = sdf_candidates[0]

        txt_file = ref_ligand.with_suffix(".txt")
        if not txt_file.exists():
            raise FileNotFoundError(f"Residue txt file not found for pocket {base}")
        pocket_ids = txt_file.read_text().split()
        yield base, pdb_file, receptor_file, ref_ligand, pocket_ids


def load_reference_ligand(ref_ligand: Path, pdb_file: Path):
    if ref_ligand.suffix.lower() == ".sdf":
        rdmol = Chem.SDMolSupplier(str(ref_ligand))[0]
        coords = torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
    else:
        structure = PDBParser(QUIET=True).get_structure("", str(pdb_file))[0]
        chain_id, resi = str(ref_ligand).split(":")
        lig_res = utils.get_residue_with_resi(structure[chain_id], int(resi))
        coords = torch.from_numpy(np.array([a.get_coord() for a in lig_res.get_atoms()])).float()

    return coords, coords.mean(axis=0), coords.shape[0]


def score_and_select(records, top_k):
    if not records:
        return []

    fps = [Chem.RDKFingerprint(rec["mol"]) for rec in records]
    distances = []
    for i, fp in enumerate(fps):
        sims = [
            DataStructs.TanimotoSimilarity(fp, other_fp)
            for j, other_fp in enumerate(fps)
            if j != i
        ]
        distances.append(0.0 if not sims else 1.0 - float(np.mean(sims)))

    def safe_value(val, default=-1e9):
        try:
            num = float(val)
        except (TypeError, ValueError):
            return default
        if math.isnan(num) or math.isinf(num):
            return default
        return num

    for rec, dist in zip(records, distances):
        metrics = rec["metrics"]
        metrics["distance"] = dist
        qed = safe_value(metrics.get("qed"))
        sa = safe_value(metrics.get("sa"))
        vina_score = safe_value(metrics.get("vina_score"))
        metrics["weighted_sum"] = 0.3 * qed + 0.1 * sa + 0.3 * dist + 0.3 * vina_score

    sorted_records = sorted(
        records, key=lambda r: r["metrics"].get("weighted_sum", -1e9), reverse=True
    )
    return sorted_records[:top_k]


def save_top_records(records, out_dir: Path, top_k: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = out_dir / f"top_{top_k}.sdf"
    utils.write_sdf_file(sdf_path, [rec["mol"] for rec in records])

    csv_path = out_dir / f"top_{top_k}_scores.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "num_atoms", "qed", "sa", "distance", "vina_score", "vina_dock", "weighted_sum"])
        for idx, rec in enumerate(records):
            m = rec["metrics"]
            writer.writerow([
                idx + 1,
                m.get("num_atoms"),
                m.get("qed"),
                m.get("sa"),
                m.get("distance"),
                m.get("vina_score"),
                m.get("vina_dock"),
                m.get("weighted_sum"),
            ])


def summarize_records(records):
    metrics = ["num_atoms", "qed", "sa", "distance", "vina_score", "vina_dock", "weighted_sum"]

    def safe_mean(values):
        values = [v for v in values if v is not None and not math.isnan(v)]
        return float(np.mean(values)) if values else None

    summary = {}
    for m in metrics:
        summary[m] = safe_mean([rec["metrics"].get(m) for rec in records])
    summary["count"] = len(records)
    return summary


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pockets = list(iter_pockets(args.dataset_dir))
    if args.limit:
        pockets = pockets[:args.limit]

    pocket_summaries = []

    for base, pdb_file, receptor_file, ref_ligand, pocket_ids in pockets:
        print(f"Processing {base}...")
        coords, mean_coord, ligand_size = load_reference_ligand(ref_ligand, pdb_file)
        num_nodes_lig = args.num_nodes_lig or ligand_size
        num_nodes_tensor = torch.ones(args.n_samples, dtype=int) * num_nodes_lig

        model = LigandPocketDDPM.load_from_checkpoint(args.checkpoint, map_location=device)
        model = model.to(device)

        reward_fn_dict = {
            "qed": model.molecule_properties.calculate_qed,
            "sa": model.molecule_properties.calculate_sa,
            "vina_score": partial(
                model.molecule_properties.calculate_docking_score,
                center_xyz=mean_coord,
                receptor_pdbqt_file=str(receptor_file),
                use_meeko=False,
                score_only=True,
            ),
            # "vina_dock": partial(
            #     model.molecule_properties.calculate_docking_score,
            #     center_xyz=mean_coord,
            #     receptor_pdbqt_file=str(receptor_file),
            #     use_meeko=False,
            #     score_only=False,
            # ),
        }

        ppo_config = SimpleNamespace(
            clip_range=args.clip_range,
            max_grad_norm=args.max_grad_norm,
            max_time_steps=model.T,
            inference_interval=args.inference_interval,
            batch_size=args.ppo_batch_size,
            lr=args.ppo_lr,
            reward_fn_dict=reward_fn_dict,
            kl_coeff_pretrain=args.kl_coeff_pretrain,
            reward_num_workers=args.reward_workers,
        )
        ppo_config.episode_length = ppo_config.max_time_steps // ppo_config.inference_interval

        if args.kl_coeff_pretrain > 0:
            model.ddpm_pretrained = copy.deepcopy(model.ddpm).to(device)
            model.ddpm_pretrained.eval()
            for param in model.ddpm_pretrained.parameters():
                param.requires_grad_(False)

        all_records = []
        for rollout_idx in range(args.rollouts):
            print(f"  Rollout {rollout_idx + 1}/{args.rollouts}...")
            _, sample_records = model.generate_ligands_rl(
                str(pdb_file),
                args.n_samples,
                pocket_ids=pocket_ids,
                ref_ligand=None if pocket_ids is not None else str(ref_ligand),
                num_nodes_lig=num_nodes_tensor,
                sanitize=args.sanitize,
                largest_frag=not args.all_frags,
                relax_iter=(200 if args.relax else 0),
                timesteps=args.timesteps,
                ppo_config=ppo_config,
                return_samples=True,
            )
            all_records.extend(sample_records)

        top_records = score_and_select(all_records, args.top_k)
        if not top_records:
            print(f"No valid molecules for {base}, skipping save.")
            continue

        pocket_out = args.output_dir / base
        save_top_records(top_records, pocket_out, args.top_k)
        print(f"Saved top {len(top_records)} ligands for {base} to {pocket_out}.")

        pocket_summary = summarize_records(top_records)
        pocket_summary["pocket"] = base
        pocket_summaries.append(pocket_summary)

    if pocket_summaries:
        summary_path = args.output_dir / "pocket_summary.csv"
        metric_fields = ["num_atoms", "qed", "sa", "distance", "vina_score", "vina_dock", "weighted_sum"]
        with open(summary_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pocket", "count"] + metric_fields)
            for row in pocket_summaries:
                writer.writerow(
                    [row["pocket"], row["count"]]
                    + [row[m] for m in metric_fields]
                )

            # Overall averages across pockets (macro average)
            overall_row = ["OVERALL", None]
            for m in metric_fields:
                values = [r[m] for r in pocket_summaries if r[m] is not None]
                overall_row.append(float(np.mean(values)) if values else None)
            writer.writerow(overall_row)

        print(f"Wrote per-pocket and overall averages to {summary_path}.")


if __name__ == "__main__":
    main()
