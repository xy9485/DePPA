#!/bin/bash
set -e

# cleanup() {
#   jobs -p | xargs -r kill
#   wait
# }

cleanup() {
  pids=$(jobs -p)
  [ -z "$pids" ] && return

  kill $pids 2>/dev/null || true
  sleep 1
  kill -9 $pids 2>/dev/null || true
  wait
}
    
# trap cleanup INT TERM
trap cleanup INT TERM EXIT
# trap 'jobs -p | xargs -r kill' INT TERM

# base_path="/home/xue/repos/DiffSBDD/rl_batch_testset_from_kis"
# group_name="${1:?Usage: $0 RESULTS_DIR}"
# results_dir="${base_path}/${group_name}"

results_dir="${1:?Usage: $0 RESULTS_DIR [N_PROCESSES]}"

last100_csv="raw_eval_last100.csv"

# [First step]
echo "run with: --top_n 100 --no_rank"
echo "Save to "$last100_csv""
python rerank_summarize_results.py --results_dir "$results_dir" --top_n 100 --no_rank &
wait

echo "First step done. Proceeding to post metrics filling..."
sleep 3

# [Second step: Post metrics filling]
post_csvs=("$last100_csv")

# Split the 0-100 span into n_processes chunks (last chunk absorbs the remainder)
n_processes="${2:-1}"
total=100
step=$(( (total + n_processes - 1) / n_processes ))  # ceil division

post_ranges=()
start=0
while [ "$start" -lt "$total" ]; do
  end=$(( start + step ))
  [ "$end" -gt "$total" ] && end=$total
  post_ranges+=("${start}-${end}")
  start=$end
done

for csv in "${post_csvs[@]}"; do
  for range in "${post_ranges[@]}"; do
    python post_metrics.py --results_dir "$results_dir" \
      --csv_name "$csv" --overwrite --skip_posecheck --dir_range "$range" &
  done
  wait
done


# # only for posecheck
# post_ranges=("0-2" "2-4" "4-6" "6-8" "8-10" "10-12" "12-14" "14-16" "16-18" "18-20" "20-22" "22-24" "24-26" "26-28" "28-30" "30-32" "32-34" "34-36" "36-38" "38-40" "40-42" "42-44" "44-46" "46-48" "48-50" "50-52" "52-54" "54-56" "56-58" "58-60" "60-62" "62-64" "64-66" "66-68" "68-70" "70-72" "72-74" "74-76" "76-78" "78-80" "80-82" "82-84" "84-86" "86-88" "88-90" "90-92" "92-94" "94-96" "96-98" "98-100")
# for csv in "${post_csvs[@]}"; do
#   for range in "${post_ranges[@]}"; do
#     python post_metrics.py --results_dir "$results_dir" \
#       --csv_name "$csv" --skip_vina --skip_diversity --skip_success_rate --skip_sc_rmsd_twoA --dir_range "$range" &
#   done
#   wait
# done