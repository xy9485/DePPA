import argparse
import csv
import math
import statistics
from pathlib import Path


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
    return round(float(sum(filtered) / len(filtered)), 6)


def safe_median(values):
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return round(float(statistics.median(filtered)), 6)


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


def summarize_mean_perPocket_overall(csv_files, save_path, fields) -> None:
    """
    This version uses csv.DictWriter.
    Aggregate per-pocket summaries and an overall mean row.
    """
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


def summarize_med_over_csv_files(csv_files, save_path, fields) -> None:
    """
    Compute medians across all rows from the provided CSV files and write a single summary row.
    """
    if not csv_files:
        raise SystemExit("No CSV files provided for summarization.")

    all_rows = []
    for csv_path in csv_files:
        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            all_rows.extend(list(reader))

    if not all_rows:
        raise SystemExit("No rows found in provided CSV files.")

    medians = {}
    for field in fields:
        values = [safe_float(row.get(field)) for row in all_rows]
        medians[field] = safe_median(values)

    overall_row = [medians[field] for field in fields]

    with save_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerow(overall_row)

    print(
        f"Wrote median statistics across {len(csv_files)} files "
        f"({len(all_rows)} rows) to {save_path}."
    )


def summarize_mean_over_csv_files(csv_files, save_path, fields) -> None:
    """
    Compute means across all rows from the provided CSV files and write a single summary row.
    """
    if not csv_files:
        raise SystemExit("No CSV files provided for summarization.")

    all_rows = []
    for csv_path in csv_files:
        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            all_rows.extend(list(reader))

    if not all_rows:
        raise SystemExit("No rows found in provided CSV files.")

    means = {}
    for field in fields:
        values = [safe_float(row.get(field)) for row in all_rows]
        means[field] = safe_mean(values)

    overall_row = [means[field] for field in fields]

    with save_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerow(overall_row)

    print(
        f"Wrote mean statistics across {len(csv_files)} files "
        f"({len(all_rows)} rows) to {save_path}."
    )


if __name__ == "__main__":
    METRIC_FIELDS = [
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
        "success_flag"
    ]
    args = parse_args()

    rerank_csv_files = []
    for pocket_dir_idx, pocket_dir in enumerate(sorted(args.results_dir.iterdir())):
        # ensure pocket_dir ends with "pocket10"
        if pocket_dir.name.endswith("pocket10"):
            rerank_csv_path = pocket_dir / args.csv_name
            if not rerank_csv_path.exists():
                # print(f"Warning: CSV file {rerank_csv_path} does not exist, skipping.")
                raise SystemExit(f"CSV file {rerank_csv_path} does not exist.")
            rerank_csv_files.append(rerank_csv_path)

    # summarize_mean_perPocket_overall(
    #     rerank_csv_files,
    #     args.results_dir / f"summary_mean_over_{args.csv_name}",
    #     fields=METRIC_FIELDS
    # )
    summarize_mean_over_csv_files(
        rerank_csv_files,
        args.results_dir / f"summary_mean_over_{args.csv_name}",
        fields=METRIC_FIELDS
    )
    summarize_med_over_csv_files(
        rerank_csv_files,
        args.results_dir / f"summary_med_over_{args.csv_name}",
        fields=METRIC_FIELDS
    )
