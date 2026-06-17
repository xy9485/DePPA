"""
Download RL training curves from W&B and plot per-group mean performance.

For each reward signal, computes the mean over runs within each group
(per policy-rollout round) and plots all group means on a single figure.

Only runs with state "finished" are included.

Usage:
    python analysis/plot_from_wandb.py \
        --project DiffSBDD-PPO \
        --entity team-yuan
"""

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

# ── Global parameters ──────────────────────────────────────────────
# GROUPS = [
#     "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_#1",
#     "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep1",
#     "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep4",
# ]

GROUPS = [
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_#1",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep1",
    # "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep2",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep3",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep4",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep5",
    # "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep7",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_clamp0.25_nmRep2",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_eval_nmRep18",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_eval_nmRep17",
]

# For the "mean_ci_settings" mode: each key is a hyperparameter setting,
# each value is the list of W&B groups (repeats) belonging to that setting.
# The mean ±1 std band is computed across the groups within each setting,
# and one curve+band is drawn per setting on a shared figure.
SETTINGS_DICT = {
    "ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_rewardNormRank": [
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_#1",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep1",
    # "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep2",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep3",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep4",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep5",
    # "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_nmRep7",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_clamp0.25_nmRep2",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_eval_nmRep18",
    "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_eval_nmRep17",
],
    "ii5_QED0.1_SA0.1_Vina0.7_Dist0.1_KL0.0_strain0.0_rewardNormZscore": [
        "ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_rewardNormzscore_kis_1",
        "ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_rewardNormzscore_kis_4",
        "ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_rewardNormzscore_kis_5",
        "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_kisReprodZscore1",
        "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_kisReprodZscore2",
        "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_kisReprodZscore3"
    ],
    "ii5_QED0.1_SA0.1_Vina0.7_Dist0.1_KL0.0_strain0.0_rewardNormMinmax": [
        "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_RTX_minmaxFixP_1",
        "nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_RTX_minmaxFixP_2",
        "Group: nIter100_nSample32_ligSizeSample_allFrags0_ii5_QED0.2_SA0.2_Vina0.5_Dist0.1_KL0.0_strain0.0_kisReprodMinmax1"
        ]
}

SIGNALS = [
    "vina_score",
    "qed",
    "sa",
    "distance",
]

STEP_KEY = "_step"


def parse_args():
    p = argparse.ArgumentParser(description="Plot per-group mean RL curves from W&B.")
    p.add_argument("--project", default="DiffSBDD-PPO")
    p.add_argument("--entity", default="team-yuan")
    p.add_argument("--base_dir", default="wandb_plots",
                   help="Base directory for saving plots.")
    p.add_argument("--output_dir", required=True,
                   help="Subfolder under --base_dir to save the plots in.")
    p.add_argument("--groups", nargs="+", default=GROUPS)
    p.add_argument("--signals", nargs="+", default=SIGNALS)
    p.add_argument("--smoothing", type=int, default=5,
                   help="Running average window size (1 = no smoothing).")
    p.add_argument("--cache_dir", type=str, default="wandb_cache",
                   help="Directory for cached group data.")
    p.add_argument("--no_cache", action="store_true",
                   help="Force re-download, ignoring cached data.")
    p.add_argument("--mode",
                   choices=["per_group", "mean_ci", "median_iqr",
                            "mean_ci_settings"],
                   default="per_group",
                   help="per_group: one curve per group + variance subplot. "
                        "mean_ci: mean across groups with ±1 std band. "
                        "median_iqr: median across groups with percentile band. "
                        "mean_ci_settings: one mean ±1 std band per setting in "
                        "SETTINGS_DICT (band is across the groups of each setting).")
    p.add_argument("--tail_steps", type=int, default=3,
                   help="Number of final steps averaged for the tail stat.")
    p.add_argument("--lower_pct", type=float, default=10.0,
                   help="Lower percentile for median_iqr band (default 25).")
    p.add_argument("--upper_pct", type=float, default=90.0,
                   help="Upper percentile for median_iqr band (default 75).")
    return p.parse_args()


def _cache_path(cache_dir, group_name):
    """Deterministic cache file path for a group."""
    h = hashlib.md5(group_name.encode()).hexdigest()[:12]
    return Path(cache_dir) / f"{h}.pkl"


def load_or_download_group(api, entity, project, group_name, signals,
                           cache_dir, use_cache=True):
    """Load cached per-run DataFrames for a group, or download from W&B.

    Returns a list of DataFrames, each with columns [STEP_KEY, *signals].
    Downloads all signals in one pass per run.
    """
    cache_file = _cache_path(cache_dir, group_name)

    if use_cache and cache_file.exists():
        print(f"  Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Download from W&B
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"group": group_name, "state": "finished"})

    keys_to_fetch = list(set([STEP_KEY] + signals))
    run_dfs = []
    for run in tqdm(list(runs), desc=f"  Downloading {group_name}", unit="run"):
        hist = run.history(keys=keys_to_fetch, pandas=True)
        if hist.empty:
            continue
        # Keep only the columns we need
        cols = [c for c in keys_to_fetch if c in hist.columns]
        run_dfs.append(hist[cols].dropna(subset=[STEP_KEY]))

    # Save to cache
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(run_dfs, f)
    print(f"  Cached {len(run_dfs)} run(s) to {cache_file}")

    return run_dfs


def group_mean_curve(run_dfs, signal):
    """Compute the per-step mean of *signal* from cached per-run DataFrames.

    Returns (steps, mean_values) as numpy arrays, or (None, None).
    """
    all_values = []
    all_steps = []
    for df in run_dfs:
        if signal not in df.columns:
            continue
        sub = df[[STEP_KEY, signal]].dropna().sort_values(STEP_KEY)
        if sub.empty:
            continue
        all_steps.append(sub[STEP_KEY].values)
        all_values.append(sub[signal].values)

    if not all_values:
        return None, None

    # Truncate to the shortest run length
    min_len = min(len(v) for v in all_values)
    print(f"  {len(all_values)} run(s) with signal '{signal}', min length: {min_len}")
    assert min_len == 100 # Expecting 100 policy-rollout rounds
    stacked = np.stack([v[:min_len] for v in all_values], axis=0)
    steps = all_steps[0][:min_len]
    mean_vals = stacked.mean(axis=0)
    return steps, mean_vals


def running_average(values, window):
    """Smooth *values* with a normalized running average of the given *window* size."""
    if window <= 1 or values.shape[0] == 0:
        return values
    K = np.ones(window)
    return np.convolve(values, K, "same") / np.convolve(np.ones(values.shape[0]), K, "same")


def setting_mean_ci(run_dfs_by_group, signal, smoothing):
    """Mean ±1 std curve across the groups of a single setting.

    Computes each group's per-step mean curve, aligns them onto a common
    step grid, then takes the mean/std across groups.

    Returns (common_steps, mean_curve, std_curve, n_groups) or None.
    """
    curves = {}  # group_name -> (steps, mean_vals)
    for g, run_dfs in run_dfs_by_group.items():
        if not run_dfs:
            continue
        steps, mean_vals = group_mean_curve(run_dfs, signal)
        if steps is None:
            continue
        curves[g] = (steps, mean_vals)

    if not curves:
        return None

    common_min = max(s.min() for s, _ in curves.values())
    common_max = min(s.max() for s, _ in curves.values())
    common_steps = np.array(sorted(set(
        int(s) for steps, _ in curves.values() for s in steps
        if common_min <= s <= common_max
    )))
    aligned = np.array([
        np.interp(common_steps, steps, vals)
        for steps, vals in curves.values()
    ])  # (n_groups, n_steps)

    mean_curve = aligned.mean(axis=0)
    std_curve = (aligned.std(axis=0, ddof=0)
                 if aligned.shape[0] > 1 else np.zeros_like(mean_curve))
    return common_steps, mean_curve, std_curve, aligned.shape[0]


def plot_mean_ci_settings(api, args, out, use_cache):
    """Plot one mean ±1 std band per hyperparameter setting in SETTINGS_DICT."""
    # Download (or load from cache) every group across all settings.
    setting_data = {}  # setting -> {group -> run_dfs}
    for setting, groups in SETTINGS_DICT.items():
        setting_data[setting] = {}
        for g in groups:
            print(f"[{setting}] Group: {g}")
            setting_data[setting][g] = load_or_download_group(
                api, args.entity, args.project, g, args.signals,
                args.cache_dir, use_cache=use_cache,
            )

    colors = plt.cm.tab10.colors

    for signal in args.signals:
        print(f"\n=== Signal: {signal} ===")
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False
        stats_lines = []  # (color, text) per setting for the summary box
        for i, setting in enumerate(SETTINGS_DICT):
            result = setting_mean_ci(
                setting_data[setting], signal, args.smoothing)
            if result is None:
                print(f"  [{setting}] no data for '{signal}'")
                continue
            common_steps, mean_curve, std_curve, n_groups = result

            center = running_average(mean_curve, args.smoothing)
            lower = running_average(mean_curve - std_curve, args.smoothing)
            upper = running_average(mean_curve + std_curve, args.smoothing)

            color = colors[i % len(colors)]
            ax.plot(common_steps, center, color=color,
                    label=f"{setting} (n={n_groups})")
            ax.fill_between(common_steps, lower, upper, color=color, alpha=0.20)
            plotted = True

            # Tail (last N steps) and middle-step statistics.
            n_tail = args.tail_steps
            tail_mean = mean_curve[-n_tail:].mean()
            tail_std = std_curve[-n_tail:].mean()
            mid = len(common_steps) // 2
            mid_step = int(common_steps[mid])
            mid_mean = mean_curve[mid]
            mid_std = std_curve[mid]
            last_step = int(common_steps[-1])
            last_mean = mean_curve[-1]
            last_std = std_curve[-1]
            print(f"  [{setting}] tail (last {n_tail}) mean={tail_mean:.3f} "
                  f"std={tail_std:.3f} | middle step {mid_step}: "
                  f"mean={mid_mean:.3f} std={mid_std:.3f} | "
                  f"last step {last_step}: mean={last_mean:.3f} "
                  f"std={last_std:.3f}")

            # Mark the middle and last steps with the mean ±1 std.
            # for m_step, m_mean, m_std in ((mid_step, mid_mean, mid_std),
            #                               (last_step, last_mean, last_std)):
            #     ax.errorbar(m_step, m_mean, yerr=m_std, color=color,
            #                 marker="o", markersize=6, capsize=4,
            #                 elinewidth=1.5, zorder=5)
            #     ax.annotate(f"{m_mean:.3g}±{m_std:.2g}",
            #                 xy=(m_step, m_mean), xytext=(4, 6),
            #                 textcoords="offset points", color=color,
            #                 fontsize=7, fontweight="bold")

            stats_lines.append((color,
                f"{setting}\n"
                f"  mid (step {mid_step}): {mid_mean:.3f}±{mid_std:.3f}\n"
                f"  last (step {last_step}): {last_mean:.3f}±{last_std:.3f}\n"
                f"  tail (last {n_tail}): {tail_mean:.3f}±{tail_std:.3f}"))

        if not plotted:
            plt.close(fig)
            print(f"Skipping signal '{signal}': no data from any setting.")
            continue

        ax.set_xlabel("Policy rollout round")
        ax.set_ylabel(signal)
        ax.set_title(f"{signal} — mean ±1 std per setting")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        # Summary box: mid / last / tail mean±std per setting.
        y = 0.97
        for color, text in stats_lines:
            ax.text(1.02, y, text, transform=ax.transAxes, color=color,
                    fontsize=7, va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor=color, alpha=0.8))
            y -= 0.02 + 0.045 * (text.count("\n") + 1)

        fig.tight_layout()
        path = (out / f"S{len(SETTINGS_DICT)}-{signal}-"
                f"smooth{args.smoothing}-meanCIsettings.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def main():
    args = parse_args()
    out = Path(args.base_dir) / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    # Download (or load from cache) all groups once
    use_cache = not args.no_cache

    if args.mode == "mean_ci_settings":
        plot_mean_ci_settings(api, args, out, use_cache)
        return

    group_data = {}  # group_name -> list of per-run DataFrames
    for g in args.groups:
        print(f"Group: {g}")
        group_data[g] = load_or_download_group(
            api, args.entity, args.project, g, args.signals,
            args.cache_dir, use_cache=use_cache,
        )
        print(f"  {len(group_data[g])} finished run(s)")

    # One plot per signal
    for signal in args.signals:
        curves = {}  # group_name -> (steps, mean_vals)
        for g in args.groups:
            run_dfs = group_data[g]
            if not run_dfs:
                continue
            steps, mean_vals = group_mean_curve(run_dfs, signal)
            if steps is None:
                print(f"  No data for signal '{signal}' in group '{g}'")
                continue
            curves[g] = (steps, mean_vals)

        if not curves:
            print(f"Skipping signal '{signal}': no data from any group.")
            continue

        # Align group means onto a common step grid (shared by all modes)
        common_min = max(s.min() for s, _ in curves.values())
        common_max = min(s.max() for s, _ in curves.values())
        print(f"Common step range for '{signal}': {common_min} to {common_max}")  # DEBUG
        common_steps = np.array(sorted(set(
            int(s) for steps, _ in curves.values() for s in steps
            if common_min <= s <= common_max
        )))
        aligned = np.array([
            np.interp(common_steps, steps, vals)
            for steps, vals in curves.values()
        ])  # (n_groups, n_steps)

        if args.mode == "per_group":
            fig, (ax, ax_var) = plt.subplots(
                2, 1, figsize=(10, 9), height_ratios=[3, 1])
            for g, (steps, mean_vals) in curves.items():
                smoothed = running_average(mean_vals, args.smoothing)
                ax.plot(steps, smoothed, label=g)

            variance = aligned.var(axis=0)
            mean_var = variance.mean()
            median_var = np.median(variance)
            max_var = variance.max()

            ax.set_xlabel("Policy rollout round")
            ax.set_ylabel(signal)
            ax.set_title(f"{signal} — mean per group")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)

            var_smooth = running_average(variance, window=1)
            ax_var.plot(common_steps, var_smooth, color="tab:red", linewidth=1.2)
            ax_var.set_xlabel("Policy rollout round")
            ax_var.set_ylabel("Variance")
            ax_var.set_title(f"{signal} — variance across groups")
            ax_var.grid(True, alpha=0.3)

            textstr = (f"Mean var:   {mean_var:.4g}\n"
                       f"Median var: {median_var:.4g}\n"
                       f"Max var:    {max_var:.4g}")
            ax_var.text(0.02, 0.95, textstr, transform=ax_var.transAxes,
                        fontsize=9, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            mode_tag = "perGroup"

        elif args.mode == "mean_ci":
            fig, ax = plt.subplots(figsize=(10, 6))
            mean_curve = aligned.mean(axis=0)
            std_curve = aligned.std(axis=0, ddof=0) if aligned.shape[0] > 1 else np.zeros_like(mean_curve)
            center = running_average(mean_curve, args.smoothing)
            lower = running_average(mean_curve - std_curve, args.smoothing)
            upper = running_average(mean_curve + std_curve, args.smoothing)
            ax.plot(common_steps, center, color="tab:blue",
                    label=f"mean over {aligned.shape[0]} groups")
            ax.fill_between(common_steps, lower, upper, color="tab:blue",
                            alpha=0.25, label="±1 std")
            ax.set_xlabel("Policy rollout round")
            ax.set_ylabel(signal)
            ax.set_title(f"{signal} — mean ±1 std across groups")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

            last_vals = aligned[:, -1]
            last_mean = last_vals.mean()
            last_std = last_vals.std(ddof=0) if last_vals.shape[0] > 1 else 0.0

            tail_mean = mean_curve[-5:].mean()
            tail_std = std_curve[-5:].mean()
            print(f"  Last 5 steps — avg mean: {tail_mean:.4g}, avg std: {tail_std:.4g}")

            textstr = (f"Last step ({int(common_steps[-1])})\n"
                       f"Mean: {last_mean:.4g}\n"
                       f"Std:  {last_std:.4g}\n"
                       f"Last 5 steps\n"
                       f"Avg mean: {tail_mean:.4g}\n"
                       f"Avg std:  {tail_std:.4g}")
            ax.text(0.02, 0.95, textstr, transform=ax.transAxes,
                    fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            mode_tag = "meanCI"

        else:  # median_iqr
            fig, ax = plt.subplots(figsize=(10, 6))
            lo_pct, hi_pct = args.lower_pct, args.upper_pct
            median_curve = np.median(aligned, axis=0)
            lo_curve = np.percentile(aligned, lo_pct, axis=0)
            hi_curve = np.percentile(aligned, hi_pct, axis=0)
            center = running_average(median_curve, args.smoothing)
            lower = running_average(lo_curve, args.smoothing)
            upper = running_average(hi_curve, args.smoothing)
            ax.plot(common_steps, center, color="tab:green",
                    label=f"median over {aligned.shape[0]} groups")
            ax.fill_between(common_steps, lower, upper, color="tab:green",
                            alpha=0.25, label=f"{lo_pct:g}–{hi_pct:g} pct")
            ax.set_xlabel("Policy rollout round")
            ax.set_ylabel(signal)
            ax.set_title(f"{signal} — median + {lo_pct:g}–{hi_pct:g} pct across groups")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            mode_tag = f"medianIQR{lo_pct:g}-{hi_pct:g}"

        fig.tight_layout()
        path = out / f"G{len(args.groups)}-{signal}-smooth{args.smoothing}-{mode_tag}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
