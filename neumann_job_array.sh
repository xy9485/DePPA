#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=sbdd_rl          # A nice readable name of your job, to see it in the queue

#SBATCH --time=1-00           # Time limit hrs:min:sec
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=32           # Number of CPUs to request
#SBATCH --gpus=1                    # Number of GPUs to request
#SBATCH --partition=ampere             # Partition to submit the job to
#SBATCH --mem=48G
#SBATCH --output=./slurm_files/slurm-%x-%A_%a.out
#SBATCH --error=./slurm_files/slurm-%x-%A_%a.err
#SBATCH --array=0-3

module load mamba
module load conda
# Activate your environment, you have to create it first
# mamba activate sbdd

module load conda
conda activate diffsbdd

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Your job script goes below this line

pocket_ranges=("0-25" "25-50" "50-75" "75-100")
# pocket_ranges=("0-2" "2-4" "4-6" "6-8")

pocket_range=${pocket_ranges[$SLURM_ARRAY_TASK_ID]}

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

# python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt --dataset_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test --sanitize --n_samples 32 --rollouts 100 --top_k 100 --mode_num_nodes_lig sample --wandb_mode online --limit $pocket_range

python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt --dataset_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test --sanitize --n_samples 32 --rollouts 100 --top_k 100 --inference_interval 10 --kl_coeff_pretrain 0.0 --mode_num_nodes_lig sample --wandb_mode online --limit $pocket_range