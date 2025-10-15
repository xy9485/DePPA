#!/bin/bash

/home/xue/repos/DiffSBDD/updateHostLaunchJSON.sh

export ENABLE_DEBUG=true

mamba activate sbdd

# Run the script with the following command:
# python -u process_crossdock.py datasets --no_H
# srun python -u /home/xue/repos/DiffSBDD/train.py --config /home/xue/repos/DiffSBDD/configs/crossdock_fullatom_joint.yml
# python -u /home/xue/repos/DiffSBDD/train.py --config /home/xue/repos/DiffSBDD/configs/crossdock_fullatom_cond.yml

python -u generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/3rfm.pdb --outfile example/3rfm_mol.sdf --ref_ligand A:330 --n_samples 20 --receptor_file example/3rfm.pdbqt
# python -u generate_ligands.py checkpoints/crossdocked_fullatom_joint.ckpt --pdbfile example/3rfm.pdb --outfile example/3rfm_mol.sdf --ref_ligand A:330 --n_samples 20

# python -u optimize.py --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/5ndu.pdb --outfile output.sdf --ref_ligand example/5ndu_C_8V2.sdf --objective sa --population_size 10 --evolution_steps 10 --top_k 10 --timesteps 100

# python -u analysis/docking.py --pdbqt_dir example/ --sdf_files example/3rfm_mol.sdf --out_dir docking_output/ --dataset crossdocked --write_csv