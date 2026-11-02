#!/bin/bash

$PROJECT/DiffSBDD_RL/updateHostLaunchJSON.sh

export ENABLE_DEBUG=true

module load miniforge3
# conda init
source "$CONDASH"
conda activate sbdd
# module load conda
# conda activate diffsbdd

# Run the script with the following command:
# python -u process_crossdock.py datasets2 --no_H
# python -u process_crossdock.py datasets2 --no_H --ca_only
# srun python -u /home/xue/repos/DiffSBDD/train.py --config /home/xue/repos/DiffSBDD/configs/crossdock_fullatom_joint.yml
# python -u /home/xue/repos/DiffSBDD/train.py --config /home/xue/repos/DiffSBDD/configs/crossdock_fullatom_cond.yml
# python -u train.py --config configs/crossdock_fullatom_cond.yml

# python -u generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/3rfm.pdb --outfile example/3rfm_mol.sdf --ref_ligand A:330 --n_samples 20 --receptor_file example/3rfm.pdbqt --sanitize
# python -u generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/5ndu.pdb --outfile example/5ndu_mol.sdf --ref_ligand A:201 --n_samples 20 --receptor_file example/5ndu.pdbqt --sanitize

# python -u generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --ref_ligand /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test/1a2g-A-rec-4jmv-1ly-lig-tt-min-0-pocket10_1a2g-A-rec-4jmv-1ly-lig-tt-min-0.sdf --pdbfile /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test/1a2g-A-rec-4jmv-1ly-lig-tt-min-0-pocket10.pdb --n_samples 32 --sanitize --mode_num_nodes_lig align_reference_ligand --all_frags

# python -u generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --ref_ligand ./datasets2/processed_crossdock_noH_full_temp/test/1d7j-A-rec-1tco-fk5-lig-tt-docked-0-pocket10_1d7j-A-rec-1tco-fk5-lig-tt-docked-0.sdf --pdbfile ./datasets2/processed_crossdock_noH_full_temp/test/1d7j-A-rec-1tco-fk5-lig-tt-docked-0-pocket10.pdb --n_samples 32 --sanitize --mode_num_nodes_lig align_reference_ligand

# python -u generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --ref_ligand datasets2/processed_crossdock_noH_full_temp/test/2gns-A-rec-4qer-stl-lig-tt-min-0-pocket10_2gns-A-rec-4qer-stl-lig-tt-min-0.sdf --pdbfile datasets2/processed_crossdock_noH_full_temp/test/2gns-A-rec-4qer-stl-lig-tt-min-0-pocket10.pdb --n_samples 32 --sanitize --mode_num_nodes_lig sample

# python -u generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --ref_ligand /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test/3daf-A-rec-3daf-feg-lig-tt-docked-0-pocket10_3daf-A-rec-3daf-feg-lig-tt-docked-0.sdf --pdbfile /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test/3daf-A-rec-3daf-feg-lig-tt-docked-0-pocket10.pdb --n_samples 32 --sanitize --mode_num_nodes_lig align_reference_ligand

# python -u generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --ref_ligand /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test/1h0i-A-rec-1e6z-ngo-lig-it2-tt-docked-15-pocket10_1h0i-A-rec-1e6z-ngo-lig-it2-tt-docked-15.sdf --pdbfile /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test/1h0i-A-rec-1e6z-ngo-lig-it2-tt-docked-15-pocket10.pdb --n_samples 32 --sanitize --mode_num_nodes_lig sample

# python -u generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --ref_ligand /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test/4z2g-A-rec-4z2g-m6v-lig-tt-docked-16-pocket10_4z2g-A-rec-4z2g-m6v-lig-tt-docked-16.sdf --pdbfile /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test/4z2g-A-rec-4z2g-m6v-lig-tt-docked-16-pocket10.pdb --n_samples 32 --sanitize --mode_num_nodes_lig sample --all_frags

# python test.py checkpoints/crossdocked_fullatom_cond.ckpt --test_dir /home/xue/repos/DiffSBDD/datasets/processed_crossdock_noH_full_temp/test --outdir crossdocked_test --sanitize --fix_n_nodes --skip_existing

# python /home/xue/repos/DiffSBDD/analysis/docking.py --pdbqt_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test --sdf_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/tewst --dataset crossdocked --out_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test_qvina_out


# python -u generate_ligands.py checkpoints/crossdocked_fullatom_joint.ckpt --pdbfile example/3rfm.pdb --outfile example/3rfm_mol.sdf --ref_ligand A:330 --n_samples 20

# python -u optimize.py --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/5ndu.pdb --outfile output.sdf --ref_ligand example/5ndu_C_8V2.sdf --objective sa --population_size 10 --evolution_steps 10 --top_k 10 --timesteps 100

# python -u analysis/docking.py --pdbqt_dir example/ --sdf_files example/3rfm_mol.sdf --out_dir docking_output/ --dataset crossdocked --write_csv

# python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt --dataset_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test --output_dir rl_batch_generate_outputs_5 --sanitize --n_samples 32 --rollouts 20 --top_k 100 --limit 10 --all_frags

# python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt --dataset_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test --output_dir rl_batch_generate_outputs_5 --sanitize --n_samples 32 --rollouts 10 --top_k 100 --limit 5:10 --mode_num_nodes_lig align_reference_ligand

# python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt --dataset_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test --sanitize --n_samples 32 --rollouts 100 --top_k 20 --inference_interval 10 --kl_coeff_pretrain 0.0 --mode_num_nodes_lig sample --wandb_mode online --limit 30-32  --group_name_suffix _test

# python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt --dataset_dir datasets2/processed_crossdock_noH_full_temp/test --sanitize --n_samples 32 --rollouts 30 --mode_num_nodes_lig sample --inference_interval 5 --kl_coeff_pretrain 0.0 --wandb_mode online --limit 30-32 --w_qed 0.2 --w_sa 0.2 --w_vina_score 0.5 --w_distance 0.1 --w_strain 0.0 --group_name_suffix _rep1_test_can_delete

# summarize pocket scores
# python3 summarize_pocket_scores.py rl_batch_testset/ligSizeRef_ii10_NormRank_Penalty-3std_QED0.3_SA0.3_Vina0.4_Distance0.0_KL0.0

# post filling metrics

# python post_metrics.py --sdf_path /home/xue/repos/DiffSBDD/rl_batch_testset/RLTestset_nRollout100_nSample32_ligSizeSample_allFrags0_ii10_NormRank_Penalty-3std_QED0.27_SA0.13_Vina0.3_Distance0.3_KL0.0/1afs-A-rec-1afs-tes-lig-tt-min-0-pocket10 --pocket_pdb_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test

# python post_metrics.py --sdf_path /home/xue/repos/DiffSBDD/rl_batch_testset/nIter20_nSample32_ligSizeSample_allFrags0_ii10_QED0.27_SA0.13_Vina0.3_Dist0.3_KL0.0_testing/3pnm-A-rec-3pnq-2ha-lig-tt-docked-10-pocket10 --pocket_pdb_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test --overwrite

# python post_metrics.py --results_dir /home/xue/repos/DiffSBDD/rl_batch_testset/RLTestset_nRollout100_nSample32_ligSizeSample_allFrags0_ii10_NormRank_Penalty-3std_QED0.27_SA0.13_Vina0.3_Distance0.3_KL0.0 --csv_name top_100_scores.csv --skip_vina

python -u batch_generate_ligands_rl.py checkpoints/crossdocked_fullatom_cond.ckpt --dataset_dir datasets2/processed_crossdock_noH_full_temp/test --sanitize --n_samples 8 --rollouts 100 --ppo_epochs 1 --n_segments 1 --mode_num_nodes_lig sample --inference_interval 5 --kl_coeff_pretrain 0.0 --wandb_mode offline --limit 30-32 --w_qed 0.2 --w_sa 0.2 --w_vina_score 0.5 --w_distance 0.1 --w_strain 0.0 --reward_norm_mode rank --group_name_suffix _kis_n64_debug