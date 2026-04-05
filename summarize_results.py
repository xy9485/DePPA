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


def compute_mean_for_single_metric_per_pocket(csv_path: Path, target_field: str, top_percentile_per_pocket: float=1.0, reverse_sort: bool=False) -> dict:
    """this function targets to compute the mean for a single specified field


    Args:
        csv_path (Path): _description_
        target_field (str): the
        top_percentile_per_pocket (float, optional): _description_. Defaults to 1.0.
        reverse_sort (bool, optional): Consider the values are passed to abs() before sorting. If True, sort in descending order. Defaults to False.

    Returns:
        dict: _description_
    """
    with csv_path.open("r", newline="") as handle:
        csv_reader = csv.DictReader(handle)
        rows = list(csv_reader)
    summary = {}
    # compute mean for each field in csv
    for field in csv_reader.fieldnames:
        if field == target_field:
            if top_percentile_per_pocket < 1.0:
                top_n_rows_to_consider = int(len(rows) * top_percentile_per_pocket)
                sorted_rows = sorted(
                    rows,
                    key=lambda row: abs(safe_float(row.get(field))),
                    reverse=reverse_sort
                )[:top_n_rows_to_consider]
                selected_rows = sorted_rows
                summary[field] = safe_mean([safe_float(row.get(field)) for row in selected_rows])
            else:
                selected_rows = rows
                summary[field] = safe_mean([safe_float(row.get(field)) for row in selected_rows])

    summary.update({"pocket": csv_path.parent.name, "count": len(selected_rows)})
    return summary

def get_median_scores(csv_path: Path) -> dict:
    with csv_path.open("r", newline="") as handle:
        csv_reader = csv.DictReader(handle)
        rows = list(csv_reader)
    summary = {"pocket": csv_path.parent.name, "count": len(rows)}
    # compute median for each field in csv
    for field in csv_reader.fieldnames:
        summary[field] = safe_median([safe_float(row.get(field)) for row in rows])

    return summary

def get_mean_scores(csv_path: Path) -> dict:
    with csv_path.open("r", newline="") as handle:
        csv_reader = csv.DictReader(handle)
        rows = list(csv_reader)
    summary = {"pocket": csv_path.parent.name, "count": len(rows)}
    # compute mean for each field in csv
    for field in csv_reader.fieldnames:
        summary[field] = safe_mean([safe_float(row.get(field)) for row in rows])
    
    # handle ourliers for vina_min
    if summary['vina_score'] < summary['vina_min']:
        summary['vina_min'] = summary['vina_score']
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

def overall_mean_singleMetric_perPocket(csv_files, save_path, target_field, top_percentile_per_pocket, reverse_sort=False) -> None:
    """
    This version uses csv.DictWriter.
    Aggregate per-pocket summaries and an overall mean row.
    """
    if not csv_files:
        raise SystemExit("No CSV files provided for summarization.")

    pocket_summaries = [
        compute_mean_for_single_metric_per_pocket(
            path,
            target_field,
            top_percentile_per_pocket,
            reverse_sort
        )
        for path in csv_files
    ]
    header = ["pocket", "count", target_field]

    # Compute the OVERALL row by averaging each metric across pockets.
    overall_row = {"pocket": "OVERALL", "count": None}
    values = [summary[target_field] for summary in pocket_summaries]
    overall_row[target_field] = safe_mean(values)

    with save_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(overall_row)
        writer.writerows(pocket_summaries)

    print(f"Wrote statistics for {len(pocket_summaries)} pockets to {save_path}.")

def summarize_results_across_pockets(csv_files, save_path, fields, mode="mean") -> None:
    """
    This version uses csv.DictWriter.
    Aggregate per-pocket summaries and an overall mean row.
    """
    if not csv_files:
        raise SystemExit("No CSV files provided for summarization.")
    if mode == "mean":
        pocket_summaries = [get_mean_scores(path) for path in csv_files]
    elif mode == "median":
        pocket_summaries = [get_median_scores(path) for path in csv_files]
    else:
        raise SystemExit(f"Unknown mode: {mode}")
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

        # values = []
        # for row in all_rows:
        #     value = safe_float(row.get(field))
        #     if field == "vina_dock":
        #         vina_score = safe_float(row.get("vina_score"))
        #         vina_min = safe_float(row.get("vina_min"))
        #         if not vina_score or not vina_min:
        #             continue
        #         if abs(vina_min) <= abs(vina_score):
        #             continue
        #         if abs(vina_score) < abs(value) and abs(vina_min) < abs(value):
        #             values.append(value)
        #         else:
        #             continue
        #     elif field == "vina_min":
        #         vina_score = safe_float(row.get("vina_score"))
        #         vina_dock = safe_float(row.get("vina_dock"))
        #         if not vina_score or not vina_dock:
        #             continue
        #         if abs(vina_score) < abs(value) <= abs(vina_dock):
        #             values.append(value)
        #         else:
        #             values.append(vina_score)
        #     else:
        #         values.append(value)
        # medians[field] = safe_median(values)
        if field == "vina_dock":
            print(f"valid vina_dock count: {len(values)} / {len(all_rows)}")
        if field == "vina_min":
            print(f"valid vina_min count: {len(values)} / {len(all_rows)}")

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
        # values = [safe_float(row.get(field)) for row in all_rows]
        # means[field] = safe_mean(values)

        values = []
        for row in all_rows:
            value = safe_float(row.get(field))
            if field == "vina_min":
                vina_score = safe_float(row.get("vina_score"))
                vina_dock = safe_float(row.get("vina_dock"))
                if not vina_score or not vina_dock:
                    continue
                if abs(vina_score) < abs(value) <= abs(vina_dock):
                    values.append(value)
                else:
                    values.append(vina_score)
            else:
                values.append(value)
        means[field] = safe_mean(values)
        if field == "vina_min":
            print(f"valid vina_min count: {len(values)} / {len(all_rows)}")

    overall_row = [means[field] for field in fields]

    with save_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerow(overall_row)

    print(
        f"Wrote mean statistics across {len(csv_files)} files "
        f"({len(all_rows)} rows) to {save_path}."
    )


def compute_percentile_given_threshold(csv_files, target_field, threshold) -> float:
    """Compute the percentile of values below a given threshold across all CSV files.

    Args:
        csv_files (list of Path): List of CSV file paths.
        target_field (str): The field to analyze.
        threshold (float): The threshold value.

    Returns:
        float: The percentile of values below the threshold.
    """
    all_values = []
    in_valid_count = 0
    for csv_path in csv_files:
        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_idx, row in enumerate(reader):
                value = safe_float(row.get(target_field))
                if value is not None:
                    all_values.append(value)
                else:
                    print(f"Invalid value for field '{target_field}' in file '{csv_path}', row {row_idx + 1}: {row.get(target_field)}")
                    in_valid_count += 1
    print(f"Total valid values for field '{target_field}': {len(all_values)}")
    print(f"Total invalid values for field '{target_field}': {in_valid_count}")
    if not all_values:
        return 0.0

    count_below_threshold = sum(1 for v in all_values if v < threshold)
    percentile = (count_below_threshold / len(all_values)) * 100.0
    percentile = round(percentile, 1)
    print(f"Values below threshold {threshold}: {count_below_threshold}")
    print(f"Percentile of values below {threshold}: {percentile}%")
    return percentile

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
        "success_flag",
        "hit_rate_flag"
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
    assert len(rerank_csv_files) == 100, f"Expected 100 CSV files, found {len(rerank_csv_files)}."

    list_top_percentile_per_pocket = [0.25, 0.5, 0.75]
    target_field = "strain"
    for top_percentile_per_pocket in list_top_percentile_per_pocket:
        overall_mean_singleMetric_perPocket(
            rerank_csv_files,
            save_path=args.results_dir / f"summary_{target_field}_top{int(top_percentile_per_pocket*100)}pct_over_{args.csv_name}",
            target_field=target_field,
            top_percentile_per_pocket=top_percentile_per_pocket,
            reverse_sort=False
        )

    summarize_results_across_pockets(
        rerank_csv_files,
        save_path=args.results_dir / f"summary_mean_over_{args.csv_name}",
        fields=METRIC_FIELDS,
        mode="mean"
    )
    summarize_results_across_pockets(
        rerank_csv_files,
        save_path=args.results_dir / f"summary_med_over_{args.csv_name}",
        fields=METRIC_FIELDS,
        mode="median"
    )
    # summarize_mean_over_csv_files(
    #     rerank_csv_files,
    #     args.results_dir / f"summary_mean_over_{args.csv_name}",
    #     fields=METRIC_FIELDS
    # )
    # summarize_med_over_csv_files(
    #     rerank_csv_files,
    #     args.results_dir / f"summary_med_over_{args.csv_name}",
    #     fields=METRIC_FIELDS
    # )

    percentile_threshold = 2.0
    target_field = "sc_rmsd"
    percentile = compute_percentile_given_threshold(
        rerank_csv_files,
        target_field=target_field,
        threshold=percentile_threshold
    )
    # Write the percentile to a text file
    percentile_file = args.results_dir / f"{target_field}_below_{percentile_threshold}_{args.csv_name}.txt"
    with percentile_file.open("w") as f:
        f.write(f"Percentile of {target_field} below {percentile_threshold}: {percentile}%\n")
    print(f"Wrote percentile information to {percentile_file}.")
