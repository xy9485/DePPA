import argparse
import csv
import math
from pathlib import Path
from rdkit import Chem

import os
if os.getenv('ENABLE_DEBUG', 'false').lower() == 'true':
    import debugpy

    # Use any open port, e.g., 5678
    debugpy.listen(("0.0.0.0", 5675))
    print("🔍 Waiting for debugger attach on port 5675...")
    debugpy.wait_for_client()

METRIC_FIELDS = [
    "num_atoms",
    "qed",
    "sa",
    "distance",
    "strain",
    "clash",
    "connectivity",
    "vina_score",
    "vina_min",
    "vina_dock",
    "sc_rmsd"
]

RERANK_WEIGHTS = {"vina_score": 5.0, "qed": 1.0, "sa": 1.5} # from paper MolJO
# RERANK_WEIGHTS = {"vina_score": 4.0, "qed": 1.0, "sa": 1.5} 
# RERANK_WEIGHTS = {"vina_score": 5.0, "qed": 1.0, "sa": 1.8} 
# RERANK_WEIGHTS = {"vina_score": 0.386, "qed": 0.186, "sa": 0.428}
# RERANK_WEIGHTS = {"vina_score": 0.39, "qed": 0.18, "sa": 0.43}
# RERANK_WEIGHTS = {"vina_score": 0.27, "qed": 0.13, "sa": 0.3, "distance": 0.3}
# RERANK_WEIGHTS = {"vina_score": 5.0, "qed": 1.0, "sa": 3.0} 
# RERANK_WEIGHTS = {"vina_score": 5.0, "qed": 1.0, "sa": 1.5, "distance": 1.0}
# RERANK_WEIGHTS = {"qed": 0.27, "sa": 0.13, "vina_score": 0.3, "distance": 0.3} 
# RERANK_WEIGHTS = {"vina_score": 5.0, "qed": 1.0, "sa": 1.5, "strain": 1.5} 


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-pocket statistics from the generated "
            "top_k score files inside an RL batch output directory."
        )
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        help="Directory that contains one subdirectory per pocket with score CSVs.",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="raw.csv",
    )
    parser.add_argument(
        "--sdf_name",
        type=str,
        default="raw.sdf",
    )
    # parser.add_argument(
    #     "--top_n_filename",
    #     type=str,
    #     default=None,
    #     help=(
    #         "Score CSV filename to load from each pocket directory. "
    #         "Pass an empty string/None to build the top list on the fly."
    #     ),
    # )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help=(
            "How many entries to keep per pocket when computing top scores "
        ),
    )
    parser.add_argument(
        "--n_range",
        type=str,
        default=None,
        help=(
            "Range of entries to select per pocket, in the format 'start:end' or 'start:'"
        ),
    )
    parser.add_argument(
        "--no_rank",
        action="store_true",
        help="If set, do not rerank; just select the top N raw entries.",
    )

    parser.add_argument(
        "--summarize_over_all_pockets",
        action="store_true",
        help="Whether to summarize over all pockets into a single CSV.",
        )
    return parser.parse_args()


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        num = float(text)
    except ValueError:
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def safe_mean(values):
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def compute_normalization_stats(rows, field_name):
    values = []
    for idx, row in enumerate(rows, start=1):
        value = safe_float(row.get(field_name))
        if value is None:
            raise SystemExit(
                f"Missing or invalid '{field_name}' in row {idx}."
            )
        values.append(value)
    if not values:
        raise SystemExit(
            f"No numeric values for '{field_name}'."
        )
    mean = float(sum(values) / len(values))
    variance = float(sum((v - mean) ** 2 for v in values) / len(values))
    std = math.sqrt(variance)
    if std == 0:
        raise SystemExit(
            f"Standard deviation for '{field_name}' is zero; "
            "cannot normalize."
        )
    return mean, std


def rerank_csv(raw_csv_path: Path, top_n: int, weights: dict) -> Path:
    with raw_csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise SystemExit(f"'{raw_csv_path}' is empty; cannot rank entries.")

    # assert keys in weights are in fieldnames of raw_csv_path
    for field in weights:
        assert field in weights, f"Field '{field}' not in weights."
    stats = {
        field: compute_normalization_stats(rows, field)
        for field in weights
    }

    for row in rows:
        score = 0.0
        for field, weight in weights.items():
            value = safe_float(row.get(field))
            if value is None:
                raise SystemExit(
                    f"Missing '{field}' value while reranking '{raw_csv_path}'."
                )
            mean, std = stats[field]
            normalized = (value - mean) / std
            score += weight * normalized
        row["z_score"] = score
        row["_z_internal"] = score

    sorted_rows = sorted(
        rows,
        key=lambda entry: entry["_z_internal"],
        reverse=True,
    )

    top_rows = sorted_rows[:top_n] if top_n > 0 else sorted_rows

    fieldnames = list(reader.fieldnames or rows[0].keys())
    if "z_score" not in fieldnames:
        fieldnames.append("z_score")

    weights_str = "_".join([f"{k}{v}" for k, v in weights.items()])

    top_path = raw_csv_path.parent / f"rerank_top{top_n}_{weights_str}.csv"
    with top_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in top_rows:
            writer.writerow(row)

    return top_path

def zscore_rerank_csv_sdf_given_dir(search_dir: Path, csv_name: str, sdf_name: str, top_n: int, weights: dict = RERANK_WEIGHTS) -> list[Path]:
    csv_path = search_dir / csv_name
    sdf_path = search_dir / sdf_name
    csv_path = csv_path.resolve()
    sdf_path = sdf_path.resolve()

    supplier = Chem.SDMolSupplier(str(sdf_path))
    molecules = []
    for idx, mol in enumerate(supplier):
        molecules.append(mol)

    with csv_path.open("r", newline="") as handle:
        csv_reader = csv.DictReader(handle)
        rows = list(csv_reader)
    assert len(rows) == len(molecules), f"len(rows) is {len(rows)} but len(molecules) is {len(molecules)}"

    for field in weights:
        assert field in csv_reader.fieldnames, f"Field '{field}' not in weights."
    stats = {
        field: compute_normalization_stats(rows, field)
        for field in weights
    }

    for row in rows:
        score = 0.0
        for field, weight in weights.items():
            value = safe_float(row.get(field))
            if value is None:
                raise SystemExit(
                    f"Missing '{field}' value while reranking '{csv_path}'."
                )
            mean, std = stats[field]
            normalized = (value - mean) / std
            score += weight * normalized
        row["weighted_sum_z_score"] = score
        row["_z_internal"] = score

    # based on weighted_sum_z_score to sort rows and molecules
    paired = list(zip(rows, molecules))
    sorted_paired = sorted(
        paired,
        key=lambda entry: entry[0]["_z_internal"],
        reverse=True,
    )

    top_paired = sorted_paired[:top_n] if top_n > 0 else sorted_paired
    top_rows = [pair[0] for pair in top_paired]
    top_molecules = [pair[1] for pair in top_paired]

    weights_str = "_".join([f"{k}{v}" for k, v in weights.items()])
    fieldnames = list(csv_reader.fieldnames or rows[0].keys())
    if "weighted_sum_z_score" not in fieldnames:
        fieldnames.append("weighted_sum_z_score")

    top_csv_path = search_dir / f"rerank_top{top_n}_{weights_str}.csv"
    with top_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in top_rows:
            writer.writerow(row)

    top_sdf_path = search_dir / f"rerank_top{top_n}_{weights_str}.sdf"
    with Chem.SDWriter(str(top_sdf_path)) as sdf_writer:
        for mol in top_molecules:
            sdf_writer.write(mol)

    return top_csv_path, top_sdf_path

def select_topN(pocket_dir: Path, csv_name, sdf_name, top_n) -> list[Path]:
    csv_path = (pocket_dir / csv_name).resolve()
    sdf_path = (pocket_dir / sdf_name).resolve()

    with csv_path.open("r", newline="") as handle:
        csv_reader = csv.DictReader(handle)
        rows = list(csv_reader)

    supplier = Chem.SDMolSupplier(str(sdf_path))
    molecules = [mol for mol in supplier]

    if not rows:
        raise SystemExit(f"'{csv_path}' is empty; cannot select entries.")
    if len(rows) != len(molecules):
        raise SystemExit(
            f"len(rows) is {len(rows)} but len(molecules) is {len(molecules)}"
        )

    top_rows = rows[-top_n:] if top_n > 0 else rows
    top_molecules = molecules[-top_n:] if top_n > 0 else molecules

    fieldnames = list(csv_reader.fieldnames or rows[0].keys())
    top_csv_path = pocket_dir / f"raw_last{top_n}.csv"
    with top_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in top_rows:
            writer.writerow(row)

    top_sdf_path = pocket_dir / f"raw_last{top_n}.sdf"
    with Chem.SDWriter(str(top_sdf_path)) as sdf_writer:
        for mol in top_molecules:
            sdf_writer.write(mol)

    return top_csv_path, top_sdf_path

def select_N_byRange(pocket_dir: Path, csv_name, sdf_name, start_idx, end_idx) -> list[Path]:
    csv_path = (pocket_dir / csv_name).resolve()
    sdf_path = (pocket_dir / sdf_name).resolve()

    with csv_path.open("r", newline="") as handle:
        csv_reader = csv.DictReader(handle)
        rows = list(csv_reader)

    supplier = Chem.SDMolSupplier(str(sdf_path))
    molecules = [mol for mol in supplier]

    if not rows:
        raise SystemExit(f"'{csv_path}' is empty; cannot select entries.")
    if len(rows) != len(molecules):
        raise SystemExit(
            f"len(rows) is {len(rows)} but len(molecules) is {len(molecules)}"
        )
    if start_idx < 0 or end_idx < 0:
        raise SystemExit(
            f"Range must be non-negative; got start_idx={start_idx}, end_idx={end_idx}."
        )
    if start_idx >= end_idx:
        raise SystemExit(
            f"Invalid range; start_idx={start_idx} must be smaller than end_idx={end_idx}."
        )
    if start_idx >= len(rows):
        raise SystemExit(
            f"start_idx={start_idx} is out of bounds for {len(rows)} entries."
        )

    bounded_end_idx = min(end_idx, len(rows))
    selected_rows = rows[start_idx:bounded_end_idx]
    selected_molecules = molecules[start_idx:bounded_end_idx]

    fieldnames = list(csv_reader.fieldnames or rows[0].keys())
    range_tag = f"{start_idx}-{bounded_end_idx}"
    selected_csv_path = pocket_dir / f"raw_range{range_tag}.csv"
    with selected_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)

    selected_sdf_path = pocket_dir / f"raw_range{range_tag}.sdf"
    with Chem.SDWriter(str(selected_sdf_path)) as sdf_writer:
        for mol in selected_molecules:
            sdf_writer.write(mol)

    return selected_csv_path, selected_sdf_path

def get_mean_scores(csv_path: Path) -> dict:
    with csv_path.open("r", newline="") as handle:
        csv_reader = csv.DictReader(handle)
        rows = list(csv_reader)
    summary = {"pocket": csv_path.parent.name, "count": len(rows)}
    # compute mean for each field in csv
    for field in csv_reader.fieldnames:
        summary[field] = safe_mean([safe_float(row.get(field)) for row in rows])

    return summary


def summarize_over_csv_files(csv_files, save_path, fields) -> None:
    pocket_summaries = [get_mean_scores(path) for path in csv_files]

    with save_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pocket", "count"] + fields)
        writer.writerow()

        overall_row = ["OVERALL", None]
        for field in fields:
            values = [row[field] for row in pocket_summaries]
            overall_row.append(safe_mean(values))
        writer.writerow(overall_row)

        for row in pocket_summaries:
            writer.writerow(
                [row["pocket"], row["count"]]
                + [row[m] for m in fields]
            )

    print(
        f"Wrote statistics for {len(pocket_summaries)} pockets to {save_path}."
    )


def summarize_over_csv_files2(csv_files, save_path, fields) -> None:
    """Aggregate per-pocket summaries and an overall mean row."""
    if not csv_files:
        raise SystemExit("No CSV files provided for summarization.")

    pocket_summaries = [get_mean_scores(path) for path in csv_files]
    header = ["pocket", "count"] + fields

    # Compute the OVERALL row by averaging each metric across pockets.
    overall_row = {"pocket": "OVERALL", "count": None}
    for field in fields:
        # values = [summary[field] for summary in pocket_summaries if summary[field] is not None]
        values = [summary.get(field, 0.0) for summary in pocket_summaries]
        overall_row[field] = safe_mean(values)

    with save_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(overall_row)
        writer.writerows(pocket_summaries)

    print(f"Wrote statistics for {len(pocket_summaries)} pockets to {save_path}.")

if __name__ == "__main__":
    # RERANK_WEIGHTS defines the ranking scheme
    args = parse_args()

    rerank_top_csv_files = []
    # iterate each subdirectory in args.results_dir to find csv files
    for pocket_dir_idx, pocket_dir in enumerate(sorted(args.results_dir.iterdir())):
        print(f"======Processing {pocket_dir_idx}th pocket directory: {pocket_dir}======")
        if not pocket_dir.is_dir():
            print(f"Skipping, not a directory.")
            continue
        # if args.no_rank and args.top_n and not args.n_range:
        #     raw_lastN_csv, raw_lastN_sdf = select_topN(pocket_dir, csv_name=args.csv_name, sdf_name=args.sdf_name, top_n=args.top_n)
        if args.no_rank and args.n_range:
            try:
                start_str, end_str = args.n_range.split(":")
                start_idx = int(start_str)
                end_idx = int(end_str)
            except Exception as e:
                raise SystemExit(f"Invalid n_range format: {args.n_range}. Expected 'start:end'. Error: {e}")
            raw_range_csv, raw_range_sdf = select_N_byRange(pocket_dir, csv_name=args.csv_name, sdf_name=args.sdf_name, start_idx=start_idx, end_idx=end_idx)
        else:
            rerank_top_csv, rerank_top_sdf = zscore_rerank_csv_sdf_given_dir(pocket_dir, csv_name=args.csv_name, sdf_name=args.sdf_name, top_n=args.top_n, weights=RERANK_WEIGHTS)
            rerank_top_csv_files.append(rerank_top_csv)

    if args.summarize_over_all_pockets:
        weights_str = "_".join([f"{k}{v}" for k, v in RERANK_WEIGHTS.items()])
        summarize_over_csv_files2(
            rerank_top_csv_files,
            args.results_dir / f"summary_rerank_top{args.top_n}_{weights_str}.csv",
            fields=METRIC_FIELDS
        )
