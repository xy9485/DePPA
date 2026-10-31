#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=sbdd_rl_2          # A nice readable name of your job, to see it in the queue

#SBATCH --time=4-00             # Time limit hrs:min:sec
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=32           # Number of CPUs to request
#SBATCH --gpus=1                    # Number of GPUs to request
#SBATCH --partition=ampere             # Partition to submit the job to
#SBATCH --mem=32G
#SBATCH --output=./slurm_files/slurm-%x-%A_%a.out
#SBATCH --error=./slurm_files/slurm-%x-%A_%a.err

module load mamba

# Activate your environment, you have to create it first
mamba activate sbdd

# Your job script goes below this line

python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --dataset_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test \
    --output_dir rl_batch_generate_outputs_minmax_P-5std_QED0.27_SA0.13_Vina0.3_Distance0.3 \
    --sanitize \
    --n_samples 32 \
    --rollouts 100 \
    --top_k 100 \
    # --limit 10 \
    # --all_frags\