import argparse
from pathlib import Path
from types import SimpleNamespace

from wander_logger import LoggerWandb 
import wandb
import torch
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages
import copy
import utils
from lightning_modules import LigandPocketDDPM
from Bio.PDB import PDBParser

import numpy as np
from rdkit import Chem
from functools import partial
import AutoDockTools
import subprocess
import os
if os.getenv('ENABLE_DEBUG', 'false').lower() == 'true':
    import debugpy

    # Use any open port, e.g., 5678
    debugpy.listen(("0.0.0.0", 5675))
    print("🔍 Waiting for debugger attach on port 5675...")
    debugpy.wait_for_client()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--pdbfile', type=str)
    parser.add_argument('--receptor_file', type=str)
    parser.add_argument('--resi_list', type=str, nargs='+', default=None)
    parser.add_argument('--ref_ligand', type=str, default=None)
    parser.add_argument('--outfile', type=Path)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_nodes_lig', type=int, default=None)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=None)
    args = parser.parse_args()

    # get pdb_id of 4 characters from the beginning of the pdbfile name
    pdb_id = Path(args.pdbfile).stem[:4]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.batch_size is None:
        args.batch_size = args.n_samples
    assert args.n_samples % args.batch_size == 0

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)


    # Identify mean coord of ligand for centering box for docking score computing, such as qvina2
    if args.ref_ligand.endswith(".sdf"):
        # ligand as sdf file
        rdmol = Chem.SDMolSupplier(str(args.ref_ligand))[0]
        ligand_coords = torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
    else:
        structure = PDBParser(QUIET=True).get_structure("", args.pdbfile)[0]
        chain_id, resi = args.ref_ligand.split(':')
        lig_res = utils.get_residue_with_resi(structure[chain_id], int(resi))
        ligand_coords = np.array([a.get_coord() for a in lig_res.get_atoms()])
    mean_coord_reference_ligand = ligand_coords.mean(axis=0)
    ligand_size = ligand_coords.shape[0]
    print(f"Reference ligand has {ligand_size} atoms.")

    if args.num_nodes_lig is not None:
        num_nodes_lig = args.num_nodes_lig
    else:
        num_nodes_lig = torch.ones(args.n_samples, dtype=int) * ligand_size
        # num_nodes_lig = None
    # RL 

    # prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')

    # tmp_folder = "/home/xue/repos/DiffSBDD/tmp"
    # tmp_pdbfile = os.path.join(tmp_folder, f"{pdb_id}.pdb")

    # # copy pdbfile to tmp folder
    # subprocess.run(["cp", args.pdbfile, tmp_folder], check=True)

    # # run prepare_receptor in tmp folder without changing global cwd
    # subprocess.run(
    #     ["python", prepare_receptor, "-r", tmp_pdbfile],
    #     check=True,
    #     cwd=tmp_folder,
    # )
    # Prefer an explicit receptor file if provided, otherwise derive from the pdbfile path.
    if args.receptor_file:
        receptor_pdbqt_file = args.receptor_file
    else:
        # Ensure we handle args.pdbfile as a path (it may be a string)
        receptor_pdbqt_file = str(Path(args.pdbfile).with_suffix('.pdbqt'))
    reward_fn_dict = {
        'qed': model.molecule_properties.calculate_qed,
        'sa': model.molecule_properties.calculate_sa,
        'vina_score': partial(model.molecule_properties.calculate_docking_score,
                          center_xyz=mean_coord_reference_ligand,
                          receptor_pdbqt_file=receptor_pdbqt_file,
                          use_meeko=False,
                          score_only=True
                          ),
        # 'vina_dock': partial(model.molecule_properties.calculate_docking_score,
        #                   center_xyz=mean_coord_reference_ligand,
        #                   receptor_pdbqt_file=receptor_pdbqt_file,
        #                   use_meeko=False,
        #                   score_only=False
        #                   ),  
    }
    ppo_config = SimpleNamespace(
        clip_range=0.2,
        max_grad_norm=0.5,
        max_time_steps=model.T,
        inference_interval=10,
        # n_samples=64,
        batch_size=32,
        # sample_n_nodes=True,
        lr=1e-5,
        # reward_fn=model.molecule_properties.calculate_qed,
        # reward_fn=model.molecule_properties.calculate_sa,
        # reward_fn=partial(model.molecule_properties.calculate_docking_score,
        #                   center_xyz=mean_coord_reference_ligand,
        #                   receptor_pdbqt_file=args.receptor_file,
        #                   use_meeko=False,
        #                   score_only=True
        #                   ),
        reward_fn_dict=reward_fn_dict,
        kl_coeff_pretrain=0.0,
    )
    wandb.init(
        project="DiffSBDD-PPO",
        mode="offline",
        group="DiffSBDD-" +pdb_id,
        name="ii10_NormRank_Penalty-3std_QED0.27_SA0.13_Vina0.3_Distance0.3_KL0.0_"+pdb_id,
        config=vars(ppo_config),
    )
    wandb_logger = LoggerWandb()

    if ppo_config.kl_coeff_pretrain > 0:
        print(f"Using KL divergence to pretrained model with coeff {ppo_config.kl_coeff_pretrain}")
        model.ddpm_pretrained = copy.deepcopy(model.ddpm).to(device)
        model.ddpm_pretrained.eval()
        for p in model.ddpm_pretrained.parameters():
            p.requires_grad_(False)
    for i in range(500):
        metrics, sample_records = model.generate_ligands_rl(
            args.pdbfile, args.batch_size, args.resi_list, args.ref_ligand,
            num_nodes_lig, args.sanitize, largest_frag=not args.all_frags,
            relax_iter=(200 if args.relax else 0),
            resamplings=args.resamplings, jump_length=args.jump_length,
            timesteps=args.timesteps, ppo_config=ppo_config)      

        metrics["General/iters"] = i
        metrics["General/timesteps"] = i * model.T

        if wandb_logger is not None:
            wandb_logger.log_and_dump(metrics)

    # molecules = []
    # for i in range(args.n_samples // args.batch_size):
    #     molecules_batch = model.generate_ligands(
    #         args.pdbfile, args.batch_size, args.resi_list, args.ref_ligand,
    #         num_nodes_lig, args.sanitize, largest_frag=not args.all_frags,
    #         relax_iter=(200 if args.relax else 0),
    #         resamplings=args.resamplings, jump_length=args.jump_length,
    #         timesteps=args.timesteps)
    #     molecules.extend(molecules_batch)

    #     # this require joint model instead of cond model
    #     # model.sample_and_analyze(n_samples=args.n_samples, dataset=None, batch_size=args.batch_size)

    # # Make SDF files
    # utils.write_sdf_file(args.outfile, molecules)
