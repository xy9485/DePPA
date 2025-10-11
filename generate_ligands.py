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

    pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.batch_size is None:
        args.batch_size = args.n_samples
    assert args.n_samples % args.batch_size == 0

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    # model has an attribute ddpm, create a same newwork like ddpm called ddpm_copy and copy the parameters
    # model.ddpm_pretrained = copy.deepcopy(model.ddpm).to(device)
    # model.ddpm_pretrained.eval()
    # for p in model.ddpm_pretrained.parameters():
    #     p.requires_grad_(False)

    if args.num_nodes_lig is not None:
        num_nodes_lig = torch.ones(args.n_samples, dtype=int) * \
                        args.num_nodes_lig
    else:
        num_nodes_lig = None

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
        reward_fn=model.molecule_properties.calculate_sa
    )
    # wandb.init(
    #     project="DiffSBDD-PPO",
    #     mode="online",
    #     group="DiffSBDD-PPO-SA",
    #     # name = run_name,
    #     config=vars(ppo_config),
    # )
    # wandb_logger = LoggerWandb()
    # for i in range(10):
    #     metrics, molecules = model.generate_ligands_rl(
    #         args.pdbfile, args.batch_size, args.resi_list, args.ref_ligand,
    #         num_nodes_lig, args.sanitize, largest_frag=not args.all_frags,
    #         relax_iter=(200 if args.relax else 0),
    #         resamplings=args.resamplings, jump_length=args.jump_length,
    #         timesteps=args.timesteps, ppo_config=ppo_config)      

    #     metrics["General/iters"] = i
    #     metrics["General/timesteps"] = i * model.T

    #     if wandb_logger is not None:
    #         wandb_logger.log_and_dump(metrics)

    molecules = []
    for i in range(args.n_samples // args.batch_size):
        molecules_batch = model.generate_ligands(
            args.pdbfile, args.batch_size, args.resi_list, args.ref_ligand,
            num_nodes_lig, args.sanitize, largest_frag=not args.all_frags,
            relax_iter=(200 if args.relax else 0),
            resamplings=args.resamplings, jump_length=args.jump_length,
            timesteps=args.timesteps)
        molecules.extend(molecules_batch)

        # this require joint model instead of cond model
        # model.sample_and_analyze(n_samples=args.n_samples, dataset=None, batch_size=args.batch_size)

    # Make SDF files
    utils.write_sdf_file(args.outfile, molecules)
