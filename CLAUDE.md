# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DiffSBDD_RL implements **DePPA (Denoising Policy Optimization)** — a framework for fine-tuning pocket-aware 3D diffusion models via RL to generate drug-like ligands in protein binding pockets. The pretrained model (DiffSBDD) is a DDPM conditioned on pocket structure, and RL training optimizes weighted rewards (QED, SA, Vina docking score, distance, strain).

## Environment Setup

```bash
./create_conda_env_torch210_cu118.sh
conda activate [ENV_NAME]
```

Key dependencies: PyTorch, PyTorch Lightning, RDKit, PyTorch Geometric, Vina/Meeko (docking), BioPython.

## Common Commands

**DePPA (RL) training — main entry point:**
```bash
python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt \
  --dataset_dir datasets2/processed_crossdock_noH_full_temp/test \
  --sanitize --n_samples 32 --rollouts 100 \
  --mode_num_nodes_lig sample --inference_interval 5 \
  --kl_coeff_pretrain 0.0 --wandb_mode offline \
  --w_qed 0.2 --w_sa 0.2 --w_vina_score 0.5 --w_distance 0.1 \
  --group_name_suffix _#1
```

**Standard diffusion model training:**
```bash
python -u train.py --config configs/crossdocked_fullatom_cond.yml
# Resume from checkpoint:
python -u train.py --config configs/crossdocked_fullatom_cond.yml --resume checkpoints/...ckpt
```

**Inference (standard, no RL):**
```bash
# Single pocket
python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
  --pdbfile example/3rfm.pdb --resi_list A:1-100 --n_samples 100

# Batch test set
python test.py checkpoints/crossdocked_fullatom_cond.ckpt \
  --test_dir datasets2/processed_crossdock_noH_full_temp/test \
  --outdir results/ --n_samples 100
```

**Receptor prep (requires `adt` conda env with Python 2.7 + MGLTools):**
```bash
conda activate adt
python analysis/docking_py27.py <pdb_dir> <pdbqt_dir> crossdocked
```

**Ligand docking (Python 3 + Meeko/QVina):**
```bash
python analysis/docking.py \
  --pdbqt_dir <receptor_pdbqt_dir> \
  --sdf_dir <ligand_sdf_dir> \
  --dataset crossdock \
  --out_dir <output_dir>
```

**SLURM array jobs:** `neumann_job_array.sh` parallelizes pocket processing across array tasks.

## Architecture

### Core Components

**`lightning_modules.py` — `LigandPocketDDPM`**
Central PyTorch Lightning module. Wraps the diffusion model, handles training/validation steps, sampling, and RL fine-tuning. Supports two modes: `pocket_conditioning` (conditional DDPM) and `joint` (joint ligand+pocket diffusion).

**`equivariant_diffusion/`**
- `conditional_model.py` — `ConditionalDDPM`: pocket-conditioned reverse diffusion, implements `sample_given_pocket()` which is the main generation method
- `en_diffusion.py` — base equivariant variational diffusion model
- `dynamics.py` — EGNN-based score network (`EGNNDynamics`)
- `egnn_new.py` — Equivariant Graph Neural Network (E(3)-equivariant message passing)

**`batch_generate_ligands_rl.py`** — RL training loop (PPO). For each pocket: rolls out `n_samples` molecules via the diffusion model, computes per-molecule rewards, normalizes via z-scoring, computes advantages, runs AdamW optimization step. Saves top-k molecules and metrics CSV per pocket.

**`dataset.py` — `ProcessedLigandPocketDataset`**
Loads pre-processed `.npz` files containing ligand-pocket pairs (coordinates, one-hot encodings, atom masks). Handles centering and optional transforms (virtual nodes).

**`analysis/`**
- `metrics.py` — molecular metrics (QED, SA)
- `molecule_builder.py` — builds RDKit molecules from coordinates + atom type one-hots
- `docking.py` — Vina/Meeko docking pipeline (Python 3)
- `docking_py27.py` — receptor PDBQT conversion using AutoDockTools (Python 2.7)

### Data Representation

Batches concatenate all graphs; `mask` tensors index into per-graph samples:
- **Ligand**: `{'x': [N,3], 'one_hot': [N,atom_types], 'size': int, 'mask': [N]}`
- **Pocket**: `{'x': [M,3], 'one_hot': [M,aa_types], 'size': int, 'mask': [M]}`

Two pocket representations are supported: `CA` (alpha-carbon only) or `full-atom`.

### Configuration

YAML configs in `configs/` (e.g., `crossdock_fullatom_cond.yml`). Key sections:
- `mode`: `pocket_conditioning` | `joint`
- `pocket_representation`: `CA` | `full-atom`
- `egnn_params`: EGNN architecture (layers, hidden dims, edge cutoffs)
- `diffusion_params`: steps, noise schedule, loss type (`l2` | `vlb`)
- `eval_params`, `wandb_params`, `auxiliary_loss`

### Key Gotchas

**`norm_values` interpretation:** `model.norm_values = [1, 4]` means coordinate scale=1, feature scale=4. The feature scale reflects the number of *unique pocket atom element types observed in the specific pocket*, not the full encoder vocabulary (10 types). This can vary pocket-to-pocket.

**Docking pipeline split:** Receptor prep requires the `adt` conda env (Python 2.7 + AutoDockTools). Ligand docking uses the main env (Python 3 + Meeko + QVina).

**RL reward normalization:** Rewards are z-scored across `n_samples` within each rollout. `--kl_coeff_pretrain` controls KL penalty to the pretrained policy for regularization.

**`--mode_num_nodes_lig`:** Use `align_reference_ligand` to match reference ligand size, or `sample` to sample from the prior.

**Virtual nodes:** Optional (`virtual_nodes: true` in config) — pads ligands to a fixed size for training stability; affects normalization/encodings.

**Debugpy support:** Set `ENABLE_DEBUG=true` environment variable to enable remote debugging.
