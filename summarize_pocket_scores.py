import argparse
import csv
import math
from pathlib import Path

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
    "vina_score",
    "vina_dock",
    "strain",
    "clash",
    "weighted_sum",
]

RERANK_WEIGHTS = {"vina_score": 5.0, "qed": 1.0, "sa": 1.5} # from paper MolJO
# RERANK_WEIGHTS = {"qed": 0.27, "sa": 0.13, "vina_score": 0.3, "distance": 0.3} # from paper MolJO


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-pocket statistics from the generated "
            "top_k score files inside an RL batch output directory."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory that contains one subdirectory per pocket with score CSVs.",
    )
    parser.add_argument(
        "--topN_filename",
        type=str,
        default=None,
        help=(
            "Score CSV filename to load from each pocket directory. "
            "Pass an empty string/None to build the top list on the fly."
        ),
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help=(
            "How many entries to keep per pocket when computing top scores "
            "internally (only used when --topN_filename is None)."
        ),
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


def compute_normalization_stats(rows, field_name, csv_path):
    values = []
    for idx, row in enumerate(rows, start=1):
        value = safe_float(row.get(field_name))
        if value is None:
            raise SystemExit(
                f"Missing or invalid '{field_name}' in '{csv_path}' row {idx}."
            )
        values.append(value)
    if not values:
        raise SystemExit(
            f"No numeric values for '{field_name}' in '{csv_path}'."
        )
    mean = float(sum(values) / len(values))
    variance = float(sum((v - mean) ** 2 for v in values) / len(values))
    std = math.sqrt(variance)
    if std == 0:
        raise SystemExit(
            f"Standard deviation for '{field_name}' in '{csv_path}' is zero; "
            "cannot normalize."
        )
    return mean, std


def rerank_raw_scores(raw_csv_path: Path, top_n: int):
    with raw_csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise SystemExit(f"'{raw_csv_path}' is empty; cannot rank entries.")

    stats = {
        field: compute_normalization_stats(rows, field, raw_csv_path)
        for field in RERANK_WEIGHTS
    }

    for row in rows:
        score = 0.0
        for field, weight in RERANK_WEIGHTS.items():
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
    assert top_n > 0
    top_rows = sorted_rows[:top_n] if top_n > 0 else sorted_rows

    fieldnames = list(reader.fieldnames or rows[0].keys())
    if "z_score" not in fieldnames:
        fieldnames.append("z_score")

    top_path = raw_csv_path.parent / f"top_{top_n}.csv"
    with top_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in top_rows:
            writer.writerow(row)

    return top_path


def build_top_score_files(output_dir: Path, top_n: int):
    if top_n <= 0:
        raise SystemExit("--top_n must be a positive integer.")

    raw_files = []
    seen = set()
    for path in output_dir.rglob("raw_scores.csv"):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        raw_files.append(path)

    raw_files.sort()
    if not raw_files:
        raise SystemExit(
            f"No 'raw_scores.csv' files found under '{output_dir}'."
        )

    generated = []
    for raw_path in raw_files:
        generated.append(rerank_raw_scores(raw_path, top_n))
    return generated


def summarize_scores(csv_path: Path):
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    summary = {"pocket": csv_path.parent.name, "count": len(rows)}
    for field in METRIC_FIELDS:
        summary[field] = safe_mean([safe_float(row.get(field)) for row in rows])
    return summary


def main():
    args = parse_args()
    output_dir: Path = args.output_dir
    if not output_dir.exists():
        raise SystemExit(f"Output directory '{output_dir}' does not exist.")

    top_filename = args.topN_filename
    if isinstance(top_filename, str) and top_filename.strip().lower() in {"", "none"}:
        top_filename = None

    if top_filename is None:
        score_files = build_top_score_files(output_dir, args.top_n)
        top_label = f"generated top_{args.top_n}.csv"
        weights_desc = "-".join(
            f"{field}{RERANK_WEIGHTS[field]}" for field in RERANK_WEIGHTS
        )
        summary_filename = f"metric_summary_top{args.top_n}_{weights_desc}.csv"
    else:
        score_files = sorted(output_dir.rglob(top_filename))
        top_label = top_filename
        summary_filename = "metric_summary.csv"
    if not score_files:
        raise SystemExit(
            f"No '{top_label}' files found under '{output_dir}'."
        )

    pocket_summaries = [summarize_scores(path) for path in score_files]

    summary_path = output_dir / summary_filename
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pocket", "count"] + METRIC_FIELDS)

        overall_row = ["OVERALL", None]
        for field in METRIC_FIELDS:
            values = [row[field] for row in pocket_summaries if row[field] is not None]
            overall_row.append(safe_mean(values))
        writer.writerow(overall_row)

        for row in pocket_summaries:
            writer.writerow(
                [row["pocket"], row["count"]]
                + [row[m] for m in METRIC_FIELDS]
            )

    print(
        f"Wrote statistics for {len(pocket_summaries)} pockets to {summary_path}."
    )


if __name__ == "__main__":
    main()
