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
import posecheck

import utils
from lightning_modules import LigandPocketDDPM

import wandb
from wander_logger import LoggerWandb

import os
if os.getenv('ENABLE_DEBUG', 'false').lower() == 'true':
    import debugpy

    # Use any open port, e.g., 5678
    debugpy.listen(("0.0.0.0", 5675))
    print("🔍 Waiting for debugger attach on port 5675...")
    debugpy.wait_for_client()

openbabel.obErrorLog.StopLogging()  # quiet OpenBabel output

METRIC_FIELDS = ["num_atoms", "qed", "sa", "distance", "vina_score", "vina_dock", "strain", "clash", "weighted_sum"]

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
        default=None,
        help="Where to store the top ligands and scores for each pocket.",
    )
    parser.add_argument("--n_samples", type=int, default=64, help="Samples per rollout.")
    parser.add_argument("--rollouts", type=int, default=10, help="Number of PPO rollouts per pocket.")
    parser.add_argument('--mode_num_nodes_lig', choices=('align_reference_ligand', 'sample'), default=None, help="Control ligand node count: keeping the same size with the reference ligand or sample conditioned on pocket size.")
    parser.add_argument("--sanitize", action="store_true", help="Sanitize generated molecules.")
    parser.add_argument("--relax", action="store_true", help="Run force field relaxation.")
    parser.add_argument("--all_frags", action="store_true", help="Keep all fragments instead of only the largest one.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override diffusion timesteps.")
    parser.add_argument("--reward_workers", type=int, default=4, help="Workers for reward computation.")
    parser.add_argument("--ppo_lr", type=float, default=1e-5, help="PPO learning rate.")
    parser.add_argument("--ppo_batch_size", type=int, default=32, help="PPO minibatch size.")
    parser.add_argument("--inference_interval", type=int, default=10, help="Steps between PPO observations.")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="PPO grad clipping.")
    parser.add_argument("--kl_coeff_pretrain", type=float, default=0.0, help="KL penalty to the pretrained policy.")
    parser.add_argument("--top_k", type=int, default=100, help="How many ligands to retain per pocket.")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="If set, generate a summary CSV aggregating per-pocket statistics after processing all pockets.",
    )
    parser.add_argument(
        "--limit",
        type=str,
        default=None,
        help=(
            "Optional index range of pockets to process. Accepts forms like '5', '2:10', or '15-20'. "
            "Single values select one index; ranges use Python-style inclusive start/exclusive end semantics."
        ),
    )
    parser.add_argument("--wandb_mode", type=str, choices=("online", "offline", "disabled"), default="online", help="wandb mode.")
    return parser.parse_args()


def resolve_index_range(limit_value, total_len):
    """Parse the --limit argument into a slice selecting pocket indices."""
    if not limit_value:
        return slice(None)

    spec = str(limit_value).strip()
    if not spec:
        return slice(None)

    def clamp_index(idx):
        if idx < 0:
            idx += total_len
        return max(0, min(total_len, idx))

    for sep in (":", "-"):
        if sep in spec:
            start_str, end_str = spec.split(sep, 1)
            # start = clamp_index(int(start_str)) if start_str else 0
            # end = clamp_index(int(end_str)) if end_str else total_len
            start = int(start_str)
            end = int(end_str)
            assert start >=0 and end >=0, "Only non-negative indices are supported."
            assert start <= end, "Start index must be less than or equal to end index."
            assert end <= total_len, "End index out of range."
            return slice(start, end)

    # idx = clamp_index(int(spec))
    idx = int(spec)
    return slice(idx, min(total_len, idx + 1))


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


def score_sort_records(records):
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
        metrics["weighted_sum"] = 0.27 * qed + 0.13 * sa + 0.3 * dist + 0.3 * vina_score

    return sorted(records, key=lambda r: r["metrics"].get("weighted_sum", -1e9), reverse=True)


def save_records(records, out_dir: Path, label: str):
    if not records:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = out_dir / f"{label}.sdf"
    utils.write_sdf_file(sdf_path, [rec["mol"] for rec in records])

    # per_record_dir = out_dir / f"top_{top_k}_individual_sdfs"
    # if per_record_dir.exists():
    #     for sdf_file in per_record_dir.glob("*.sdf"):
    #         try:
    #             sdf_file.unlink()
    #         except FileNotFoundError:
    #             pass
    # else:
    #     per_record_dir.mkdir()

    csv_path = out_dir / f"{label}_scores.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank"] + METRIC_FIELDS)
        for idx, rec in enumerate(records):
            m = rec["metrics"]
            writer.writerow([
                idx + 1,
            ] + [m.get(field) for field in METRIC_FIELDS])

            # mol = rec.get("mol")
            # if mol is None:
            #     continue
            # for key, value in m.items():
            #     if value is None:
            #         continue
            #     mol.SetProp(str(key), str(value))
            # per_record_sdf = per_record_dir / f"rank_{idx + 1:03d}.sdf"
            # utils.write_sdf_file(per_record_sdf, [mol])


def summarize_records(records):
    # metrics = ["num_atoms", "qed", "sa", "distance", "vina_score", "vina_dock", "strain", "clash", "weighted_sum"]

    def safe_mean(values):
        values = [v for v in values if v is not None and not math.isnan(v)]
        return float(np.mean(values)) if values else None

    summary = {}
    for m in METRIC_FIELDS:
        summary[m] = safe_mean([rec["metrics"].get(m) for rec in records])
    summary["count"] = len(records)
    return summary


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # args.output_dir.mkdir(parents=True, exist_ok=True)

    pockets = list(iter_pockets(args.dataset_dir))
    if args.limit is not None:
        try:
            index_slice = resolve_index_range(args.limit, len(pockets))
        except ValueError as exc:
            raise SystemExit(f"Invalid --limit value '{args.limit}': {exc}") from exc
        # start_idx = 0 if index_slice.start is None else index_slice.start
        # stop_idx = len(pockets) if index_slice.stop is None else index_slice.stop
        assert index_slice.start is not None and index_slice.stop is not None, "Only non-negative indices are supported."
        pockets = pockets[index_slice]
        print(f"Restricting to pocket indices {index_slice.start}:{index_slice.stop} ({len(pockets)} total)")

    pocket_summaries = []

    for pocket_idx, (base, pdb_file, receptor_file, ref_ligand, pocket_ids) in enumerate(pockets):
        print(f"Processing {base}...Pocket {pocket_idx + 1}...Total Pockets: {len(pockets)}")
        coords, mean_coord, ligand_size = load_reference_ligand(ref_ligand, pdb_file)
        # num_nodes_lig = args.num_nodes_lig or ligand_size
        if args.mode_num_nodes_lig == 'align_reference_ligand':
            num_nodes_lig = torch.ones(args.n_samples, dtype=int) * ligand_size
        elif args.mode_num_nodes_lig == 'sample' or args.mode_num_nodes_lig is None:
            # num_nodes_lig = None leads to sampling num nodes conditioned on pocket size
            num_nodes_lig = None

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
            # 'posecheck': partial(model.molecule_properties.pose_check,
            #             posecheck_protein=posecheck.utils.loading.load_protein_from_pdb(pdb_file),
            #             compute_strain=True,
            #             compute_clash=True,
            #             compute_interactions=False
            #             ),
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
            reward_weights={
                "qed": 0.27,
                "sa": 0.13,
                "vina_score": 0.3,
                "distance": 0.3,
            },
        )
        ppo_config.episode_length = ppo_config.max_time_steps // ppo_config.inference_interval

        if args.kl_coeff_pretrain > 0:
            model.ddpm_pretrained = copy.deepcopy(model.ddpm).to(device)
            model.ddpm_pretrained.eval()
            for param in model.ddpm_pretrained.parameters():
                param.requires_grad_(False)

        ligSize = "Ref" if args.mode_num_nodes_lig == 'align_reference_ligand' else "Sample"
        group_name = f"RLTestset_nRollout_{args.rollouts}_nSample{args.n_samples}_ligSize{ligSize}_allFrags{int(args.all_frags)}_ii{args.inference_interval}_NormRank_Penalty-3std_QED{ppo_config.reward_weights['qed']}_SA{ppo_config.reward_weights['sa']}_Vina{ppo_config.reward_weights['vina_score']}_Distance{ppo_config.reward_weights['distance']}_KL{args.kl_coeff_pretrain}"
        wandb.init(
            project="DiffSBDD-PPO",
            mode=args.wandb_mode,
            group=group_name,
            name=" "+Path(pdb_file).stem[:4],
            config=vars(ppo_config),
        )
        wandb_logger = LoggerWandb()
        print("ppo_config:", ppo_config)
    
        if args.output_dir:
            output_dir = Path("rl_batch_testset") / args.output_dir
        else:
            output_dir = Path("rl_batch_testset") / Path(group_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            all_records = []
            for rollout_idx in range(args.rollouts):
                print(f"  Rollout {rollout_idx + 1}/{args.rollouts}...")
                episodic_metrics, sample_records = model.generate_ligands_rl(
                    str(pdb_file),
                    args.n_samples,
                    pocket_ids=pocket_ids,
                    ref_ligand=None if pocket_ids is not None else str(ref_ligand),
                    num_nodes_lig=num_nodes_lig,
                    sanitize=args.sanitize,
                    largest_frag=not args.all_frags,
                    relax_iter=(200 if args.relax else 0),
                    timesteps=args.timesteps,
                    ppo_config=ppo_config,
                    return_samples=True,
                )
                all_records.extend(sample_records)
                episodic_metrics["General/rollout_idx"] = rollout_idx
                if wandb_logger is not None:
                    wandb_logger.log_and_dump(episodic_metrics)

            if not all_records:
                print(f"No molecules generated for {base}, skipping save.")
                continue

            pocket_out = output_dir / base
            save_records(all_records, pocket_out, "raw")

            scored_records = score_sort_records(all_records)
            if not scored_records:
                print(f"No valid molecules for {base} after scoring, skipping save.")
                continue

            save_records(scored_records, pocket_out, "all_sorted")

            top_records = scored_records[:args.top_k]
            save_records(top_records, pocket_out, f"top_{args.top_k}")
            print(f"Saved top {len(top_records)} ligands for {base} to {pocket_out}.")

            pocket_summary = summarize_records(top_records)
            pocket_summary["pocket"] = base
            pocket_summaries.append(pocket_summary)
        finally:
            wandb.finish()

    if args.summary:
        summary_path = output_dir / "pocket_summary.csv"
        # metric_fields = ["num_atoms", "qed", "sa", "distance", "vina_score", "vina_dock", "strain", "clash", "weighted_sum"]
        with open(summary_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pocket", "count"] + METRIC_FIELDS)
            for row in pocket_summaries:
                writer.writerow(
                    [row["pocket"], row["count"]]
                    + [row[m] for m in METRIC_FIELDS]
                )

            # Overall averages across pockets (macro average)
            overall_row = ["OVERALL", None]
            for m in METRIC_FIELDS:
                values = [r[m] for r in pocket_summaries if r[m] is not None]
                overall_row.append(float(np.mean(values)) if values else None)
            writer.writerow(overall_row)

        print(f"Wrote per-pocket and overall averages to {summary_path}.")


if __name__ == "__main__":
    main()
