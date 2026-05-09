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

METRIC_FIELDS = ["num_atoms", "qed", "sa", "distance", "vina_score", "vina_min", "vina_dock", "strain", "clash", "connectivity", "sc_rmsd"]

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
    parser.add_argument("--group_name_suffix", type=str, default="", help="Suffix to append to the wandb group name.")
    parser.add_argument("--slurm_job_tag", type=str, default="", help="Tag to identify the SLURM job, appended to wandb group name.")
    parser.add_argument("--w_qed", type=float, default=0.27, help="Weight for qed in the reward function.")
    parser.add_argument("--w_sa", type=float, default=0.13, help="Weight for sa in the reward function.")
    parser.add_argument("--w_vina_score", type=float, default=0.3, help="Weight for vina_score in the reward function.")
    parser.add_argument("--w_distance", type=float, default=0.3, help="Weight for distance in the reward function.")
    parser.add_argument("--w_strain", type=float, default=0.0, help="Weight for strain in the reward function.")
    parser.add_argument(
        "--vina_center_xyz_mode",
        type=str,
        choices=("lig", "ref"),
        default="ref",
        help="Docking box center: generated ligand center ('lig') or reference ligand center ('ref').",
    )
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


def compute_diversity(records):
    fps = [Chem.RDKFingerprint(rec["mol"]) for rec in records]
    distances = []
    for i, fp in enumerate(fps):
        sims = [
            DataStructs.TanimotoSimilarity(fp, other_fp)
            for j, other_fp in enumerate(fps)
            if j != i
        ]
        distances.append(0.0 if not sims else 1.0 - float(np.mean(sims)))
    return distances


def safe_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.item()
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def compute_normalization_stats_from_records(records, field_name):
    values = []
    for idx, rec in enumerate(records, start=1):
        metrics = rec.get("metrics", {})
        value = safe_float(metrics.get(field_name))
        if value is None:
            raise ValueError(
                f"Missing or invalid '{field_name}' metric in record {idx}."
            )
        values.append(value)
    if not values:
        raise ValueError(f"No numeric values for '{field_name}'.")
    mean = float(sum(values) / len(values))
    variance = float(sum((v - mean) ** 2 for v in values) / len(values))
    std = math.sqrt(variance)
    if std == 0:
        raise ValueError(
            f"Standard deviation for '{field_name}' is zero; cannot normalize."
        )
    return mean, std


def sort_records_by_zscore(records, weights):
    """Sort records via z-scored weighted sum, mirroring rerank_summarize_results."""
    if not records:
        return []

    stats = {
        field: compute_normalization_stats_from_records(records, field)
        for field in weights
    }

    for rec in records:
        score = 0.0
        metrics = rec.get("metrics", {})
        for field, weight in weights.items():
            value = safe_float(metrics.get(field))
            if value is None:
                raise ValueError(
                    f"Missing '{field}' metric while computing rerank z-score."
                )
            mean, std = stats[field]
            normalized = (value - mean) / std
            score += weight * normalized
        metrics["z_score_weighted_sum"] = score

    return sorted(
        records,
        key=lambda rec: rec["metrics"].get("z_score_weighted_sum", 0.0),
        reverse=True,
    )


def weighted_score(rec, weights):
    metrics = rec["metrics"]
    weighted_score = 0.0
    for key, weight in weights.items():
        value = metrics[key]
        weighted_score += weight * value
    return weighted_score


def save_records(records, out_dir: Path, label: str,):
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

    csv_path = out_dir / f"{label}.csv"
    # [using csv.writer]
    # csv_fields = list(records[0]["metrics"].keys())
    # with open(csv_path, "w", newline="") as handle:
    #     writer = csv.writer(handle)
    #     writer.writerow(["rank"] + csv_fields)
    #     for idx, rec in enumerate(records):
    #         m = rec["metrics"]
    #         writer.writerow([
    #             idx + 1,
    #         ] + [m.get(field) for field in csv_fields])

    # [using csv.Dictwriter instead of csv.writer]
    csv_fields = list(records[0]["metrics"].keys())
    csv_fields = ['rank',] + csv_fields
    with open(csv_path, "w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        csv_writer.writeheader()
        for idx, rec in enumerate(records):
            row_dict = {"rank": idx + 1}
            row_dict.update(rec["metrics"])
            csv_writer.writerow(row_dict)

            # mol = rec.get("mol")
            # if mol is None:
            #     continue
            # for key, value in m.items():
            #     if value is None:
            #         continue
            #     mol.SetProp(str(key), str(value))
            # per_record_sdf = per_record_dir / f"rank_{idx + 1:03d}.sdf"
            # utils.write_sdf_file(per_record_sdf, [mol])
    return sdf_path, csv_path


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
        vina_center_xyz = None if args.vina_center_xyz_mode == "lig" else mean_coord
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
                center_xyz=vina_center_xyz,
                receptor_pdbqt_file=str(receptor_file),
                use_meeko=False,
                vina_mode='vina_score',
            ),
            # "vina_min": partial(
            #     model.molecule_properties.calculate_docking_score,
            #     center_xyz=vina_center_xyz,
            #     receptor_pdbqt_file=str(receptor_file),
            #     use_meeko=False,
            #     vina_mode='vina_min',
            # ),
            # "vina_dock": partial(
            #     model.molecule_properties.calculate_docking_score,
            #     center_xyz=mean_coord,
            #     receptor_pdbqt_file=str(receptor_file),
            #     use_meeko=False,
            #     vina_mode='vina_dock',
            # ),
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
        reward_weights={
                "qed": args.w_qed,
                "sa": args.w_sa,
                "vina_score": args.w_vina_score,
                "distance": args.w_distance,
                "strain": args.w_strain
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
            reward_weights=reward_weights,
        )
        ppo_config.episode_length = ppo_config.max_time_steps // ppo_config.inference_interval

        if args.kl_coeff_pretrain > 0:
            model.ddpm_pretrained = copy.deepcopy(model.ddpm).to(device)
            model.ddpm_pretrained.eval()
            for param in model.ddpm_pretrained.parameters():
                param.requires_grad_(False)

        ligSize = "Ref" if args.mode_num_nodes_lig == 'align_reference_ligand' else "Sample"
        group_name = f"nIter{args.rollouts}_nSample{args.n_samples}_ligSize{ligSize}_allFrags{int(args.all_frags)}_ii{args.inference_interval}_QED{ppo_config.reward_weights['qed']}_SA{ppo_config.reward_weights['sa']}_Vina{ppo_config.reward_weights['vina_score']}_Dist{ppo_config.reward_weights['distance']}_KL{args.kl_coeff_pretrain}_strain{ppo_config.reward_weights.get('strain', 0.0)}"
        group_name += args.group_name_suffix
        assert len(group_name) <= 128, "WandB group name exceeds 128 characters."

        if args.output_dir:
            output_dir = Path("records") / args.output_dir
        else:
            output_dir = Path("records") / Path(group_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Skip pockets whose outputs already exist (e.g. after Slurm requeue/preemption).
        pocket_out = output_dir / base
        if (pocket_out / "raw.csv").exists():
            print(f"  Skipping {base}: outputs already exist at {pocket_out}")
            continue

        wandb.init(
            project="DiffSBDD-PPO",
            mode=args.wandb_mode,
            group=group_name,
            name=f"{Path(pdb_file).stem[:4]}-{args.slurm_job_tag}",
            config=vars(ppo_config),
        )
        wandb_logger = LoggerWandb()
        print("ppo_config:", ppo_config)

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
                # Eval
                if rollout_idx == args.rollouts - 1:
                    ppo_config_eval = copy.deepcopy(ppo_config)
                    ppo_config_eval.inference_interval = 1
                    all_records_eval = []
                    all_episodic_metrics_eval = []

                    for eval_iter in range(4):
                        with torch.no_grad():
                            episodic_metrics_eval, sample_records_eval = model.generate_ligands_rl(
                                str(pdb_file),
                                25, # 25 samples for eval iteration
                                pocket_ids=pocket_ids,
                                ref_ligand=None if pocket_ids is not None else str(ref_ligand),
                                num_nodes_lig=num_nodes_lig,
                                # n_nodes_min=1,
                                sanitize=args.sanitize,
                                largest_frag=not args.all_frags,
                                relax_iter=(200 if args.relax else 0),
                                timesteps=args.timesteps,
                                ppo_config=ppo_config_eval,
                                policy_update=False,
                                return_samples=True,
                            )
                        all_records_eval.extend(sample_records_eval)
                        all_episodic_metrics_eval.append(episodic_metrics_eval)
                    # prefix "Eval/" to metric keys for eval logging
                    wandb_table_cols = ["eval_iter"] + list(episodic_metrics_eval.keys())
                    eval_table = wandb.Table(columns=wandb_table_cols)
                    for eval_iter, episodic_metrics_eval in enumerate(all_episodic_metrics_eval):
                        eval_table.add_data(eval_iter, *episodic_metrics_eval.values())
                    wandb_logger.log_and_dump({"Eval/table": eval_table})
                    # episodic_metrics_eval = {f"Eval/{k}": v for k, v in episodic_metrics_eval.items()}
                    # episodic_metrics_eval["General/rollout_idx"] = rollout_idx
                    # if wandb_logger is not None:
                        # wandb_logger.log_and_dump(episodic_metrics_eval)


            if not all_records:
                print(f"No molecules generated for {base}, skipping save.")
                continue

            # distance2 = compute_diversity(all_records)
            # # distance2 is computed among all_records, distance computed among among one rollout
            # for rec, dist in zip(all_records, distance2):
            #     rec["metrics"]["distance2"] = dist

            pocket_out = output_dir / base
            save_records(all_records, pocket_out, "raw")
            save_records(all_records_eval, pocket_out, "raw_eval")

            ckpt_path = pocket_out / "model.ckpt"
            torch.save(
                {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)},
                ckpt_path,
            )
            print(f"Saved model checkpoint to {ckpt_path}")
            
            # [recording processed results with weighted sum for sorting]
            # weights_for_sorting = reward_weights.copy()
            # sorted_records = sorted(
            #     all_records,
            #     key=partial(weighted_score, weights=weights_for_sorting),
            #     reverse=True,
            # )

            # save_records(sorted_records, pocket_out, "sorted")

            # # score_records2 follows rerank_summarize_results z-scored sorting
            # zscore_sorted_records = sort_records_by_zscore(all_records, {"vina_score": 5.0, "qed": 1.0, "sa": 1.5})
            # save_records(zscore_sorted_records, pocket_out, "zscore_sorted")

            # top_records = zscore_sorted_records[:args.top_k]
            # distance2 = compute_diversity(top_records)
            # # distance2 is computed among all_records, distance computed among among one rollout
            # for rec, dist in zip(top_records, distance2):
            #     rec["metrics"]["distance2"] = dist
            # save_records(top_records, pocket_out, f"top_{args.top_k}")
            # print(f"Saved top {len(top_records)} ligands for {base} to {pocket_out}.")

            # pocket_summary = summarize_records(top_records)
            # pocket_summary["pocket"] = base
            # pocket_summaries.append(pocket_summary)
        finally:
            wandb.finish()

    # if args.summary:
    #     summary_path = output_dir / "pocket_summary.csv"
    #     # metric_fields = ["num_atoms", "qed", "sa", "distance", "vina_score", "vina_dock", "strain", "clash", "weighted_sum"]
    #     with open(summary_path, "w", newline="") as handle:
    #         writer = csv.writer(handle)
    #         writer.writerow(["pocket", "count"] + METRIC_FIELDS)
    #         for row in pocket_summaries:
    #             writer.writerow(
    #                 [row["pocket"], row["count"]]
    #                 + [row[m] for m in METRIC_FIELDS]
    #             )

    #         # Overall averages across pockets (macro average)
    #         overall_row = ["OVERALL", None]
    #         for m in METRIC_FIELDS:
    #             values = [r[m] for r in pocket_summaries if r[m] is not None]
    #             overall_row.append(float(np.mean(values)) if values else None)
    #         writer.writerow(overall_row)

    #     print(f"Wrote per-pocket and overall averages to {summary_path}.")


if __name__ == "__main__":
    main()
    print("All Done.")
