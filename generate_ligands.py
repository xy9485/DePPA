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
from equivariant_diffusion.s_predictor import SPredictor, EGNNSPredictor

import numpy as np

from functools import partial

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
    parser.add_argument('--s_predictor_mode', type=str, choices=['mlp', 'egnn', None], default='egnn',
                        help="Which s-predictor to attach after loading the checkpoint: 'mlp' (SPredictor), 'egnn' (EGNNSPredictor), or 'none'.")
    args = parser.parse_args()

    pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.batch_size is None:
        args.batch_size = args.n_samples
    assert args.n_samples % args.batch_size == 0

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    # Attach invariant s-predictor AFTER loading (keeps checkpoint compatibility)
    model.s_predictor = None
    if args.s_predictor_mode == 'mlp':
        try:
            sp = SPredictor(atom_nf=model.atom_nf, residue_nf=model.aa_nf, n_dims=model.x_dims).to(device)
            model.ddpm.s_predictor = sp  # attach without touching the saved state_dict
        except Exception as e:
            print(f"[s-predictor] Failed to attach SPredictor: {e}. Continuing without s-predictor.")
            model.ddpm.s_predictor = None
    elif args.s_predictor_mode == 'egnn':
        try:
            atom_encoder = copy.deepcopy(model.ddpm.dynamics.atom_encoder)
            residue_encoder = copy.deepcopy(model.ddpm.dynamics.residue_encoder)
            backbone = copy.deepcopy(model.ddpm.dynamics.egnn)
            sp = EGNNSPredictor(atom_encoder=atom_encoder, 
                                residue_encoder=residue_encoder, 
                                backbone=backbone,
                                n_dims=model.x_dims, 
                                h_nf=model.ddpm.dynamics.node_nf,
                                condition_time=True,
                                update_pocket_coords=False, 
                                edge_cutoff_ligand=model.ddpm.dynamics.edge_cutoff_l,
                                edge_cutoff_pocket=model.ddpm.dynamics.edge_cutoff_p,
                                edge_cutoff_interaction=model.ddpm.dynamics.edge_cutoff_i,
                                device=device).to(device)
            model.ddpm.s_predictor = sp
        except Exception as e:
            print(f"[s-predictor] Failed to attach EGNNSPredictor: {e}. Continuing without s-predictor.")
            model.ddpm.s_predictor = None
    else:
        # args.s_predictor == 'none'
        model.ddpm.s_predictor = None

    # model has an attribute ddpm, create a same network like ddpm called ddpm_copy and copy the parameters
    # model.ddpm_pretrained = copy.deepcopy(model.ddpm).to(device)
    # model.ddpm_pretrained.eval()
    # for p in model.ddpm_pretrained.parameters():
    #     p.requires_grad_(False)

    if args.num_nodes_lig is not None:
        num_nodes_lig = torch.ones(args.n_samples, dtype=int) * \
                        args.num_nodes_lig
    else:
        num_nodes_lig = None

    # Identify mean coord of ligand for centering box for docking score computing, such as qvina2
    structure = PDBParser(QUIET=True).get_structure("", args.pdbfile)[0]
    chain_id, resi = args.ref_ligand.split(':')
    lig_res = utils.get_residue_with_resi(structure[chain_id], int(resi))
    coords = np.array([a.get_coord() for a in lig_res.get_atoms()])
    mean_coord_reference_ligand = coords.mean(axis=0)

    # RL 
    ppo_config = SimpleNamespace(
        clip_range=0.2,
        max_grad_norm=0.5,
        max_time_steps=model.T,
        inference_interval=10,
        n_samples=64,
        batch_size=5,
        sample_n_nodes=True,
        lr=1e-5,
        # reward_fn=model.molecule_properties.calculate_qed,
        # reward_fn=model.molecule_properties.calculate_sa,
        reward_fn=partial(model.molecule_properties.calculate_docking_score,
                          center_xyz=mean_coord_reference_ligand,
                          receptor_pdbqt_file=args.receptor_file,
                          use_meeko=False
                          ),
        predict_s=False if model.ddpm.s_predictor is None else True,
    )
    run_name = "predict_s"
    wandb.init(
        project="DiffSBDD-PPO",
        mode="online",
        group="DiffSBDD-PPO-DockingScore",
        name=run_name if run_name is not None else None,
        config=vars(ppo_config),
    )
    wandb_logger = LoggerWandb()
    for i in range(600):
        metrics, molecules = model.generate_ligands_rl(
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
