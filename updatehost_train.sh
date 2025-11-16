#!/bin/bash

/home/xue/repos/DiffSBDD/updateHostLaunchJSON.sh

export ENABLE_DEBUG=true

mamba activate sbdd
# module load conda
# conda activate diffsbdd

# Run the script with the following command:
# python -u process_crossdock.py datasets2 --no_H
# python -u process_crossdock.py datasets2 --no_H --ca_only
# srun python -u /home/xue/repos/DiffSBDD/train.py --config /home/xue/repos/DiffSBDD/configs/crossdock_fullatom_joint.yml
# python -u /home/xue/repos/DiffSBDD/train.py --config /home/xue/repos/DiffSBDD/configs/crossdock_fullatom_cond.yml

# python -u generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/3rfm.pdb --outfile example/3rfm_mol.sdf --ref_ligand A:330 --n_samples 20 --receptor_file example/3rfm.pdbqt --sanitize
# python -u generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/5ndu.pdb --outfile example/5ndu_mol.sdf --ref_ligand A:201 --n_samples 20 --receptor_file example/5ndu.pdbqt --sanitize

python -u generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt  --ref_ligand /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test/1a2g-A-rec-4jmv-1ly-lig-tt-min-0-pocket10_1a2g-A-rec-4jmv-1ly-lig-tt-min-0.sdf --pdbfile /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test/1a2g-A-rec-4jmv-1ly-lig-tt-min-0-pocket10.pdb --n_samples 20 --sanitize 

# python test.py checkpoints/crossdocked_fullatom_cond.ckpt --test_dir /home/xue/repos/DiffSBDD/datasets/processed_crossdock_noH_full_temp/test --outdir crossdocked_test --sanitize --fix_n_nodes --skip_existing

# python /home/xue/repos/DiffSBDD/analysis/docking.py --pdbqt_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test --sdf_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test --dataset crossdocked --out_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test_qvina_out


# python -u generate_ligands.py checkpoints/crossdocked_fullatom_joint.ckpt --pdbfile example/3rfm.pdb --outfile example/3rfm_mol.sdf --ref_ligand A:330 --n_samples 20

# python -u optimize.py --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/5ndu.pdb --outfile output.sdf --ref_ligand example/5ndu_C_8V2.sdf --objective sa --population_size 10 --evolution_steps 10 --top_k 10 --timesteps 100

# python -u analysis/docking.py --pdbqt_dir example/ --sdf_files example/3rfm_mol.sdf --out_dir docking_output/ --dataset crossdocked --write_csv