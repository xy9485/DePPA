#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=diffsbdd
#SBATCH -t 10:00:00
#SBATCH -p kisski               # kisski or kisski-h100
#SBATCH --gres=gpu:A100:1            # gpu:A100:1 or gpu:H100:1
#SBATCH --output=./slurm_files/slurm-%x-%A_%a.out
#SBATCH --error=./slurm_files/slurm-%x-%A_%a.err
#SBATCH --mem=40G
#SBATCH --array=0-3
##SBATCH --chdir=/user/yuan.xue/u17632/DiffSBDD_RL
#SBATCH --chdir=/projects/extern/kisski/kisski-l3s-rl4diff/dir.project/DiffSBDD_RL
#SBATCH --constraint=inet

# set -euo pipefail

# Use a job-specific temp dir so module/conda init does not fail when /tmp is full on a node.
# export TMPDIR="${HOME}/tmp/${SLURM_JOB_ID:-$$}"
# export TMPDIR="${HOME}/tmp"
# mkdir -p "$TMPDIR"
# trap 'rm -rf "$TMPDIR"' EXIT

source ~/.bashrc
# module load mamba
module load miniforge3
# Activate your environment, you have to create it first
# mamba activate sbdd

# conda activate diffsbdd
conda activate sbdd

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Your job script goes below this line
# pocket_ranges=("0-25" "25-50")
# pocket_ranges=("0-25" "25-50" "50-75" "75-100")
# pocket_ranges=("0-10" "10-20" "20-30" "30-40" "40-50" "50-60" "60-70" "70-80" "80-90" "90-100")
# pocket_ranges=("0-2" "2-4" "4-6" "6-8")
# pocket_ranges=("0-2" "25-27" "50-52" "75-77")

# pocket_range=${pocket_ranges[$SLURM_ARRAY_TASK_ID]}

# Total pockets to split across the job array. The range for this task is
# derived from the array size (#SBATCH --array above), so changing the array
# range automatically re-chunks the work — no need to edit a hardcoded list.
total_pockets=100
n_tasks=${SLURM_ARRAY_TASK_COUNT:-1}
chunk=$(( (total_pockets + n_tasks - 1) / n_tasks ))   # ceil(total / n_tasks)
start=$(( SLURM_ARRAY_TASK_ID * chunk ))
end=$(( start + chunk ))
(( end > total_pockets )) && end=$total_pockets
pocket_range="${start}-${end}"

array_job_id=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
# Track both the parent array job and the specific task for logging.
job_tag="${array_job_id}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    job_tag="${job_tag}_${SLURM_ARRAY_TASK_ID}"
fi

# python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt \
#     --dataset_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test \
#     --output_dir rl_batch_generate_outputs_minmax_P-5std_QED0.27_SA0.13_Vina0.3_Distance0.3 \
#     --sanitize \
#     --n_samples 32 \
#     --rollouts 100 \
#     --top_k 100 \
#     --limit  \
#     # --all_frags\

# python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt --dataset_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test --sanitize --output_dir test3 --n_samples 32 --rollouts 3 --top_k 10 --mode_num_nodes_lig sample --wandb_mode online --limit $pocket_range 

# srun -u python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt --dataset_dir datasets2/processed_crossdock_noH_full_temp/test --sanitize --n_samples 64 --rollouts 100 --mode_num_nodes_lig sample --inference_interval 5 --kl_coeff_pretrain 0.0 --wandb_mode online --limit $pocket_range --w_qed 0.2 --w_sa 0.2 --w_vina_score 0.5 --w_distance 0.1 --w_strain 0.0 --reward_norm_mode rank --slurm_job_tag ${job_tag} --group_name_suffix _kis_n64_15

# srun -u python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt --dataset_dir datasets2/processed_crossdock_noH_full_temp/test --sanitize --n_samples 32 --rollouts 100 --mode_num_nodes_lig align_reference_ligand --inference_interval 5 --kl_coeff_pretrain 0.05 --wandb_mode online --limit $pocket_range --w_qed 0.27 --w_sa 0.13 --w_vina_score 0.3 --w_distance 0.3 --w_strain 0.0 --all_frags --slurm_job_tag ${job_tag} --group_name_suffix _kl.05_nodes+-1#1

# python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt --dataset_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test --sanitize --n_samples 32 --rollouts 6 --top_k 20 --inference_interval 10 --kl_coeff_pretrain 0.0 --mode_num_nodes_lig sample --wandb_mode online --limit $pocket_range --group_name_suffix _test

srun -u python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt --dataset_dir datasets2/processed_crossdock_noH_full_temp/test --sanitize --n_samples 32 --rollouts 100 --ppo_epochs 1 --n_segments 50 --amp_dtype bf16 --mode_num_nodes_lig sample --inference_interval 5 --kl_coeff_pretrain 0.0 --wandb_mode online --limit $pocket_range --w_qed 0.2 --w_sa 0.2 --w_vina_score 0.5 --w_distance 0.1 --w_strain 0.0 --reward_norm_mode rank --slurm_job_tag ${job_tag} --group_name_suffix _#1