from collections import defaultdict
import math
from argparse import Namespace
from typing import Optional
from time import time
from pathlib import Path
from concurrent import futures
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
from torch_scatter import scatter_add, scatter_mean
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one

from constants import dataset_params, FLOAT_TYPE, INT_TYPE
from equivariant_diffusion.dynamics import EGNNDynamics
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from equivariant_diffusion.conditional_model import ConditionalDDPM, \
    SimpleConditionalDDPM
from dataset import ProcessedLigandPocketDataset
import utils
from analysis.visualization import save_xyz_file, visualize, visualize_chain
from analysis.metrics import BasicMolecularMetrics, CategoricalDistribution, \
    MoleculeProperties
from analysis.molecule_builder import build_molecule, process_molecule
from analysis.docking import smina_score

from types import SimpleNamespace
from rdkit import Chem, DataStructs
from scipy.stats import norm
from functools import partial

class LigandPocketDDPM(pl.LightningModule):
    def __init__(
            self,
            outdir,
            dataset,
            datadir,
            batch_size,
            lr,
            egnn_params: Namespace,
            diffusion_params,
            num_workers,
            augment_noise,
            augment_rotation,
            clip_grad,
            eval_epochs,
            eval_params,
            visualize_sample_epoch,
            visualize_chain_epoch,
            auxiliary_loss,
            loss_params,
            mode,
            node_histogram,
            pocket_representation='CA',
            virtual_nodes=False
    ):
        super(LigandPocketDDPM, self).__init__()
        self.save_hyperparameters()

        ddpm_models = {'joint': EnVariationalDiffusion,
                       'pocket_conditioning': ConditionalDDPM,
                       'pocket_conditioning_simple': SimpleConditionalDDPM}
        assert mode in ddpm_models
        self.mode = mode
        assert pocket_representation in {'CA', 'full-atom'}
        self.pocket_representation = pocket_representation

        self.dataset_name = dataset
        self.datadir = datadir
        self.outdir = outdir
        self.batch_size = batch_size
        self.eval_batch_size = eval_params.eval_batch_size \
            if 'eval_batch_size' in eval_params else batch_size
        self.lr = lr
        self.loss_type = diffusion_params.diffusion_loss_type
        self.eval_epochs = eval_epochs
        self.visualize_sample_epoch = visualize_sample_epoch
        self.visualize_chain_epoch = visualize_chain_epoch
        self.eval_params = eval_params
        self.num_workers = num_workers
        self.augment_noise = augment_noise
        self.augment_rotation = augment_rotation
        self.dataset_info = dataset_params[dataset]
        self.T = diffusion_params.diffusion_steps
        self.clip_grad = clip_grad
        if clip_grad:
            self.gradnorm_queue = utils.Queue()
            # Add large value that will be flushed.
            self.gradnorm_queue.add(3000)

        self.lig_type_encoder = self.dataset_info['atom_encoder']
        self.lig_type_decoder = self.dataset_info['atom_decoder']
        self.pocket_type_encoder = self.dataset_info['aa_encoder'] \
            if self.pocket_representation == 'CA' \
            else self.dataset_info['atom_encoder']
        self.pocket_type_decoder = self.dataset_info['aa_decoder'] \
            if self.pocket_representation == 'CA' \
            else self.dataset_info['atom_decoder']

        smiles_list = None if eval_params.smiles_file is None \
            else np.load(eval_params.smiles_file)
        self.ligand_metrics = BasicMolecularMetrics(self.dataset_info,
                                                    smiles_list)
        self.molecule_properties = MoleculeProperties()
        self.ligand_type_distribution = CategoricalDistribution(
            self.dataset_info['atom_hist'], self.lig_type_encoder)
        if self.pocket_representation == 'CA':
            self.pocket_type_distribution = CategoricalDistribution(
                self.dataset_info['aa_hist'], self.pocket_type_encoder)
        else:
            self.pocket_type_distribution = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.virtual_nodes = virtual_nodes
        self.data_transform = None
        self.max_num_nodes = len(node_histogram) - 1
        if virtual_nodes:
            # symbol = 'virtual'
            symbol = 'Ne'  # visualize as Neon atoms
            self.lig_type_encoder[symbol] = len(self.lig_type_encoder)
            self.virtual_atom = self.lig_type_encoder[symbol]
            self.lig_type_decoder.append(symbol)
            self.data_transform = utils.AppendVirtualNodes(
                self.max_num_nodes, self.lig_type_encoder, symbol)

            # Update dataset_info dictionary. This is necessary for using the
            # visualization functions.
            self.dataset_info['atom_encoder'] = self.lig_type_encoder
            self.dataset_info['atom_decoder'] = self.lig_type_decoder

        self.atom_nf = len(self.lig_type_decoder)
        self.aa_nf = len(self.pocket_type_decoder)
        self.x_dims = 3

        net_dynamics = EGNNDynamics(
            atom_nf=self.atom_nf,
            residue_nf=self.aa_nf,
            n_dims=self.x_dims,
            joint_nf=egnn_params.joint_nf,
            device=egnn_params.device if torch.cuda.is_available() else 'cpu',
            hidden_nf=egnn_params.hidden_nf,
            act_fn=torch.nn.SiLU(),
            n_layers=egnn_params.n_layers,
            attention=egnn_params.attention,
            tanh=egnn_params.tanh,
            norm_constant=egnn_params.norm_constant,
            inv_sublayers=egnn_params.inv_sublayers,
            sin_embedding=egnn_params.sin_embedding,
            normalization_factor=egnn_params.normalization_factor,
            aggregation_method=egnn_params.aggregation_method,
            edge_cutoff_ligand=egnn_params.__dict__.get('edge_cutoff_ligand'),
            edge_cutoff_pocket=egnn_params.__dict__.get('edge_cutoff_pocket'),
            edge_cutoff_interaction=egnn_params.__dict__.get('edge_cutoff_interaction'),
            update_pocket_coords=(self.mode == 'joint'),
            reflection_equivariant=egnn_params.reflection_equivariant,
            edge_embedding_dim=egnn_params.__dict__.get('edge_embedding_dim'),
        )

        self.ddpm = ddpm_models[self.mode](
                dynamics=net_dynamics,
                atom_nf=self.atom_nf,
                residue_nf=self.aa_nf,
                n_dims=self.x_dims,
                timesteps=diffusion_params.diffusion_steps,
                noise_schedule=diffusion_params.diffusion_noise_schedule,
                noise_precision=diffusion_params.diffusion_noise_precision,
                loss_type=diffusion_params.diffusion_loss_type,
                norm_values=diffusion_params.normalize_factors,
                size_histogram=node_histogram,
                virtual_node_idx=self.lig_type_encoder[symbol] if virtual_nodes else None
        )

        self.auxiliary_loss = auxiliary_loss
        self.lj_rm = self.dataset_info['lennard_jones_rm']
        if self.auxiliary_loss:
            self.clamp_lj = loss_params.clamp_lj
            self.auxiliary_weight_schedule = WeightSchedule(
                T=diffusion_params.diffusion_steps,
                max_weight=loss_params.max_weight, mode=loss_params.schedule)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.ddpm.parameters(), lr=self.lr,
                                 amsgrad=True, weight_decay=1e-12)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'train.npz'), transform=self.data_transform)
            self.val_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'val.npz'), transform=self.data_transform)
        elif stage == 'test':
            self.test_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'test.npz'), transform=self.data_transform)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self.train_dataset.collate_fn,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.val_dataset.collate_fn,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.test_dataset.collate_fn,
                          pin_memory=True)

    def get_ligand_and_pocket(self, data):
        ligand = {
            'x': data['lig_coords'].to(self.device, FLOAT_TYPE),
            'one_hot': data['lig_one_hot'].to(self.device, FLOAT_TYPE),
            'size': data['num_lig_atoms'].to(self.device, INT_TYPE),
            'mask': data['lig_mask'].to(self.device, INT_TYPE),
        }
        if self.virtual_nodes:
            ligand['num_virtual_atoms'] = data['num_virtual_atoms'].to(
                self.device, INT_TYPE)

        pocket = {
            'x': data['pocket_coords'].to(self.device, FLOAT_TYPE),
            'one_hot': data['pocket_one_hot'].to(self.device, FLOAT_TYPE),
            'size': data['num_pocket_nodes'].to(self.device, INT_TYPE),
            'mask': data['pocket_mask'].to(self.device, INT_TYPE)
        }
        return ligand, pocket

    def forward(self, data):
        ligand, pocket = self.get_ligand_and_pocket(data)

        # Note: \mathcal{L} terms in the paper represent log-likelihoods while
        # our loss terms are a negative(!) log-likelihoods
        delta_log_px, error_t_lig, error_t_pocket, SNR_weight, \
        loss_0_x_ligand, loss_0_x_pocket, loss_0_h, neg_log_const_0, \
        kl_prior, log_pN, t_int, xh_lig_hat, info = \
            self.ddpm(ligand, pocket, return_info=True)

        if self.loss_type == 'l2' and self.training:
            actual_ligand_size = ligand['size'] - ligand['num_virtual_atoms'] if self.virtual_nodes else ligand['size']

            # normalize loss_t
            denom_lig = self.x_dims * actual_ligand_size + \
                        self.ddpm.atom_nf * ligand['size']
            error_t_lig = error_t_lig / denom_lig
            denom_pocket = (self.x_dims + self.ddpm.residue_nf) * pocket['size']
            error_t_pocket = error_t_pocket / denom_pocket
            loss_t = 0.5 * (error_t_lig + error_t_pocket)

            # normalize loss_0
            loss_0_x_ligand = loss_0_x_ligand / (self.x_dims * actual_ligand_size)
            loss_0_x_pocket = loss_0_x_pocket / (self.x_dims * pocket['size'])
            loss_0 = loss_0_x_ligand + loss_0_x_pocket + loss_0_h

        # VLB objective or evaluation step
        else:
            # Note: SNR_weight should be negative
            loss_t = -self.T * 0.5 * SNR_weight * (error_t_lig + error_t_pocket)
            loss_0 = loss_0_x_ligand + loss_0_x_pocket + loss_0_h
            loss_0 = loss_0 + neg_log_const_0

        nll = loss_t + loss_0 + kl_prior

        # Correct for normalization on x.
        if not (self.loss_type == 'l2' and self.training):
            nll = nll - delta_log_px

            # always the same number of nodes if virtual nodes are added
            if not self.virtual_nodes:
                # Transform conditional nll into joint nll
                # Note:
                # loss = -log p(x,h|N) and log p(x,h,N) = log p(x,h|N) + log p(N)
                # Therefore, log p(x,h|N) = -loss + log p(N)
                # => loss_new = -log p(x,h,N) = loss - log p(N)
                nll = nll - log_pN

        # Add auxiliary loss term
        if self.auxiliary_loss and self.loss_type == 'l2' and self.training:
            x_lig_hat = xh_lig_hat[:, :self.x_dims]
            h_lig_hat = xh_lig_hat[:, self.x_dims:]
            weighted_lj_potential = \
                self.auxiliary_weight_schedule(t_int.long()) * \
                self.lj_potential(x_lig_hat, h_lig_hat, ligand['mask'])
            nll = nll + weighted_lj_potential
            info['weighted_lj'] = weighted_lj_potential.mean(0)

        info['error_t_lig'] = error_t_lig.mean(0)
        info['error_t_pocket'] = error_t_pocket.mean(0)
        info['SNR_weight'] = SNR_weight.mean(0)
        info['loss_0'] = loss_0.mean(0)
        info['kl_prior'] = kl_prior.mean(0)
        info['delta_log_px'] = delta_log_px.mean(0)
        info['neg_log_const_0'] = neg_log_const_0.mean(0)
        info['log_pN'] = log_pN.mean(0)
        return nll, info

    def lj_potential(self, atom_x, atom_one_hot, batch_mask):
        adj = batch_mask[:, None] == batch_mask[None, :]
        adj = adj ^ torch.diag(torch.diag(adj))  # remove self-edges
        edges = torch.where(adj)

        # Compute pair-wise potentials
        r = torch.sum((atom_x[edges[0]] - atom_x[edges[1]])**2, dim=1).sqrt()

        # Get optimal radii
        lennard_jones_radii = torch.tensor(self.lj_rm, device=r.device)
        # unit conversion pm -> A
        lennard_jones_radii = lennard_jones_radii / 100.0
        # normalization
        lennard_jones_radii = lennard_jones_radii / self.ddpm.norm_values[0]
        atom_type_idx = atom_one_hot.argmax(1)
        rm = lennard_jones_radii[atom_type_idx[edges[0]],
                                 atom_type_idx[edges[1]]]
        sigma = 2 ** (-1 / 6) * rm
        out = 4 * ((sigma / r) ** 12 - (sigma / r) ** 6)

        if self.clamp_lj is not None:
            out = torch.clamp(out, min=None, max=self.clamp_lj)

        # Compute potential per atom
        out = scatter_add(out, edges[0], dim=0, dim_size=len(atom_x))

        # Sum potentials of all atoms
        return scatter_add(out, batch_mask, dim=0)

    def log_metrics(self, metrics_dict, split, batch_size=None, **kwargs):
        for m, value in metrics_dict.items():
            self.log(f'{m}/{split}', value, batch_size=batch_size, **kwargs)

    def training_step(self, data, *args):
        if self.augment_noise > 0:
            raise NotImplementedError
            # Add noise eps ~ N(0, augment_noise) around points.
            # eps = sample_center_gravity_zero_gaussian(x.size(), x.device)
            # x = x + eps * args.augment_noise

        if self.augment_rotation:
            raise NotImplementedError
            # x = utils.random_rotation(x).detach()

        try:
            nll, info = self.forward(data)
        except RuntimeError as e:
            # this is not supported for multi-GPU
            if self.trainer.num_devices < 2 and 'out of memory' in str(e):
                print('WARNING: ran out of memory, skipping to the next batch')
                return None
            else:
                raise e

        loss = nll.mean(0)

        info['loss'] = loss
        self.log_metrics(info, 'train', batch_size=len(data['num_lig_atoms']))

        return info

    def _shared_eval(self, data, prefix, *args):
        nll, info = self.forward(data)
        loss = nll.mean(0)

        info['loss'] = loss

        self.log_metrics(info, prefix, batch_size=len(data['num_lig_atoms']),
                         sync_dist=True)

        return info

    def validation_step(self, data, *args):
        self._shared_eval(data, 'val', *args)

    def test_step(self, data, *args):
        self._shared_eval(data, 'test', *args)

    def on_validation_epoch_end(self, validation_step_outputs):

        # Perform validation on single GPU
        if not self.trainer.is_global_zero:
            return

        suffix = '' if self.mode == 'joint' else '_given_pocket'

        if (self.current_epoch + 1) % self.eval_epochs == 0:
            tic = time()

            sampling_results = getattr(self, 'sample_and_analyze' + suffix)(
                self.eval_params.n_eval_samples, self.val_dataset,
                batch_size=self.eval_batch_size)
            self.log_metrics(sampling_results, 'val')

            print(f'Evaluation took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_sample_epoch == 0:
            tic = time()
            getattr(self, 'sample_and_save' + suffix)(
                self.eval_params.n_visualize_samples)
            print(f'Sample visualization took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_chain_epoch == 0:
            tic = time()
            getattr(self, 'sample_chain_and_save' + suffix)(
                self.eval_params.keep_frames)
            print(f'Chain visualization took {time() - tic:.2f} seconds')

    @torch.no_grad()
    def sample_and_analyze(self, n_samples, dataset=None, batch_size=None):
        print(f'Analyzing sampled molecules at epoch {self.current_epoch}...')

        batch_size = self.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, n_samples)

        # each item in molecules is a tuple (position, atom_type_encoded)
        molecules = []
        atom_types = []
        aa_types = []
        for i in range(math.ceil(n_samples / batch_size)):

            n_samples_batch = min(batch_size, n_samples - len(molecules))

            num_nodes_lig, num_nodes_pocket = \
                self.ddpm.size_distribution.sample(n_samples_batch)

            xh_lig, xh_pocket, lig_mask, _ = self.ddpm.sample(
                n_samples_batch, num_nodes_lig, num_nodes_pocket,
                device=self.device)

            x = xh_lig[:, :self.x_dims].detach().cpu()
            atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()
            lig_mask = lig_mask.cpu()

            molecules.extend(list(
                zip(utils.batch_to_list(x, lig_mask),
                    utils.batch_to_list(atom_type, lig_mask))
            ))

            atom_types.extend(atom_type.tolist())
            aa_types.extend(
                xh_pocket[:, self.x_dims:].argmax(1).detach().cpu().tolist())

        return self.analyze_sample(molecules, atom_types, aa_types)

    def analyze_sample(self, molecules, atom_types, aa_types, receptors=None):
        # Distribution of node types
        kl_div_atom = self.ligand_type_distribution.kl_divergence(atom_types) \
            if self.ligand_type_distribution is not None else -1
        kl_div_aa = self.pocket_type_distribution.kl_divergence(aa_types) \
            if self.pocket_type_distribution is not None else -1

        # Convert into rdmols
        rdmols = [build_molecule(*graph, self.dataset_info) for graph in molecules]

        # Other basic metrics
        (validity, connectivity, uniqueness, novelty), (_, connected_mols) = \
            self.ligand_metrics.evaluate_rdmols(rdmols)

        qed, sa, logp, lipinski, diversity = \
            self.molecule_properties.evaluate_mean(connected_mols)

        out = {
            'kl_div_atom_types': kl_div_atom,
            'kl_div_residue_types': kl_div_aa,
            'Validity': validity,
            'Connectivity': connectivity,
            'Uniqueness': uniqueness,
            'Novelty': novelty,
            'QED': qed,
            'SA': sa,
            'LogP': logp,
            'Lipinski': lipinski,
            'Diversity': diversity
        }

        # Simple docking score
        if receptors is not None:
            # out['smina_score'] = np.mean(smina_score(rdmols, receptors))
            out['smina_score'] = np.mean(smina_score(connected_mols, receptors))

        return out

    def get_full_path(self, receptor_name):
        pdb, suffix = receptor_name.split('.')
        receptor_name = f'{pdb.upper()}-{suffix}.pdb'
        return Path(self.datadir, 'val', receptor_name)

    @torch.no_grad()
    def sample_and_analyze_given_pocket(self, n_samples, dataset=None,
                                        batch_size=None):
        print(f'Analyzing sampled molecules given pockets at epoch '
              f'{self.current_epoch}...')

        batch_size = self.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, n_samples)

        # each item in molecules is a tuple (position, atom_type_encoded)
        molecules = []
        atom_types = []
        aa_types = []
        receptors = []
        for i in range(math.ceil(n_samples / batch_size)):

            n_samples_batch = min(batch_size, n_samples - len(molecules))

            # Create a batch
            batch = dataset.collate_fn(
                [dataset[(i * batch_size + j) % len(dataset)]
                 for j in range(n_samples_batch)]
            )

            ligand, pocket = self.get_ligand_and_pocket(batch)
            receptors.extend([self.get_full_path(x) for x in batch['receptors']])

            if self.virtual_nodes:
                num_nodes_lig = self.max_num_nodes
            else:
                num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
                    n1=None, n2=pocket['size'])

            xh_lig, xh_pocket, lig_mask, _ = self.ddpm.sample_given_pocket(
                pocket, num_nodes_lig)

            x = xh_lig[:, :self.x_dims].detach().cpu()
            atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()
            lig_mask = lig_mask.cpu()

            if self.virtual_nodes:
                # Remove virtual nodes for analysis
                vnode_mask = (atom_type == self.virtual_atom)
                x = x[~vnode_mask, :]
                atom_type = atom_type[~vnode_mask]
                lig_mask = lig_mask[~vnode_mask]

            molecules.extend(list(
                zip(utils.batch_to_list(x, lig_mask),
                    utils.batch_to_list(atom_type, lig_mask))
            ))

            atom_types.extend(atom_type.tolist())
            aa_types.extend(
                xh_pocket[:, self.x_dims:].argmax(1).detach().cpu().tolist())

        return self.analyze_sample(molecules, atom_types, aa_types,
                                   receptors=receptors)

    def sample_and_save(self, n_samples):
        num_nodes_lig, num_nodes_pocket = \
            self.ddpm.size_distribution.sample(n_samples)

        xh_lig, xh_pocket, lig_mask, pocket_mask = \
            self.ddpm.sample(n_samples, num_nodes_lig, num_nodes_pocket,
                             device=self.device)

        if self.pocket_representation == 'CA':
            # convert residues into atom representation for visualization
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                xh_pocket[:, :self.x_dims], self.lig_type_encoder)
        else:
            x_pocket, one_hot_pocket = \
                xh_pocket[:, :self.x_dims], xh_pocket[:, self.x_dims:]
        x = torch.cat((xh_lig[:, :self.x_dims], x_pocket), dim=0)
        one_hot = torch.cat((xh_lig[:, self.x_dims:], one_hot_pocket), dim=0)

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}')
        save_xyz_file(str(outdir) + '/', one_hot, x, self.lig_type_decoder,
                      name='molecule',
                      batch_mask=torch.cat((lig_mask, pocket_mask)))
        # visualize(str(outdir), dataset_info=self.dataset_info, wandb=wandb)
        visualize(str(outdir), dataset_info=self.dataset_info, wandb=None)

    def sample_and_save_given_pocket(self, n_samples):
        batch = self.val_dataset.collate_fn(
            [self.val_dataset[i] for i in torch.randint(len(self.val_dataset),
                                                        size=(n_samples,))]
        )
        ligand, pocket = self.get_ligand_and_pocket(batch)

        if self.virtual_nodes:
            num_nodes_lig = self.max_num_nodes
        else:
            num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])

        xh_lig, xh_pocket, lig_mask, pocket_mask = \
            self.ddpm.sample_given_pocket(pocket, num_nodes_lig)

        if self.pocket_representation == 'CA':
            # convert residues into atom representation for visualization
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                xh_pocket[:, :self.x_dims], self.lig_type_encoder)
        else:
            x_pocket, one_hot_pocket = \
                xh_pocket[:, :self.x_dims], xh_pocket[:, self.x_dims:]
        x = torch.cat((xh_lig[:, :self.x_dims], x_pocket), dim=0)
        one_hot = torch.cat((xh_lig[:, self.x_dims:], one_hot_pocket), dim=0)

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}')
        save_xyz_file(str(outdir) + '/', one_hot, x, self.lig_type_decoder,
                      name='molecule',
                      batch_mask=torch.cat((lig_mask, pocket_mask)))
        # visualize(str(outdir), dataset_info=self.dataset_info, wandb=wandb)
        visualize(str(outdir), dataset_info=self.dataset_info, wandb=None)

    def sample_chain_and_save(self, keep_frames):
        n_samples = 1

        num_nodes_lig, num_nodes_pocket = \
            self.ddpm.size_distribution.sample(n_samples)

        chain_lig, chain_pocket, _, _ = self.ddpm.sample(
            n_samples, num_nodes_lig, num_nodes_pocket,
            return_frames=keep_frames, device=self.device)

        chain_lig = utils.reverse_tensor(chain_lig)
        chain_pocket = utils.reverse_tensor(chain_pocket)

        # Repeat last frame to see final sample better.
        chain_lig = torch.cat([chain_lig, chain_lig[-1:].repeat(10, 1, 1)],
                              dim=0)
        chain_pocket = torch.cat(
            [chain_pocket, chain_pocket[-1:].repeat(10, 1, 1)], dim=0)

        # Prepare entire chain.
        x_lig = chain_lig[:, :, :self.x_dims]
        one_hot_lig = chain_lig[:, :, self.x_dims:]
        one_hot_lig = F.one_hot(
            torch.argmax(one_hot_lig, dim=2),
            num_classes=len(self.lig_type_decoder))
        x_pocket = chain_pocket[:, :, :self.x_dims]
        one_hot_pocket = chain_pocket[:, :, self.x_dims:]
        one_hot_pocket = F.one_hot(
            torch.argmax(one_hot_pocket, dim=2),
            num_classes=len(self.pocket_type_decoder))

        if self.pocket_representation == 'CA':
            # convert residues into atom representation for visualization
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                x_pocket, self.lig_type_encoder)

        x = torch.cat((x_lig, x_pocket), dim=1)
        one_hot = torch.cat((one_hot_lig, one_hot_pocket), dim=1)

        # flatten (treat frame (chain dimension) as batch for visualization)
        x_flat = x.view(-1, x.size(-1))
        one_hot_flat = one_hot.view(-1, one_hot.size(-1))
        mask_flat = torch.arange(x.size(0)).repeat_interleave(x.size(1))

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}', 'chain')
        save_xyz_file(str(outdir), one_hot_flat, x_flat, self.lig_type_decoder,
                      name='/chain', batch_mask=mask_flat)
        visualize_chain(str(outdir), self.dataset_info, wandb=wandb)

    def sample_chain_and_save_given_pocket(self, keep_frames):
        n_samples = 1

        batch = self.val_dataset.collate_fn([
            self.val_dataset[torch.randint(len(self.val_dataset), size=(1,))]
        ])
        ligand, pocket = self.get_ligand_and_pocket(batch)

        if self.virtual_nodes:
            num_nodes_lig = self.max_num_nodes
        else:
            num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])

        chain_lig, chain_pocket, _, _ = self.ddpm.sample_given_pocket(
            pocket, num_nodes_lig, return_frames=keep_frames)

        chain_lig = utils.reverse_tensor(chain_lig)
        chain_pocket = utils.reverse_tensor(chain_pocket)

        # Repeat last frame to see final sample better.
        chain_lig = torch.cat([chain_lig, chain_lig[-1:].repeat(10, 1, 1)],
                              dim=0)
        chain_pocket = torch.cat(
            [chain_pocket, chain_pocket[-1:].repeat(10, 1, 1)], dim=0)

        # Prepare entire chain.
        x_lig = chain_lig[:, :, :self.x_dims]
        one_hot_lig = chain_lig[:, :, self.x_dims:]
        one_hot_lig = F.one_hot(
            torch.argmax(one_hot_lig, dim=2),
            num_classes=len(self.lig_type_decoder))
        x_pocket = chain_pocket[:, :, :3]
        one_hot_pocket = chain_pocket[:, :, 3:]
        one_hot_pocket = F.one_hot(
            torch.argmax(one_hot_pocket, dim=2),
            num_classes=len(self.pocket_type_decoder))

        if self.pocket_representation == 'CA':
            # convert residues into atom representation for visualization
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                x_pocket, self.lig_type_encoder)

        x = torch.cat((x_lig, x_pocket), dim=1)
        one_hot = torch.cat((one_hot_lig, one_hot_pocket), dim=1)

        # flatten (treat frame (chain dimension) as batch for visualization)
        x_flat = x.view(-1, x.size(-1))
        one_hot_flat = one_hot.view(-1, one_hot.size(-1))
        mask_flat = torch.arange(x.size(0)).repeat_interleave(x.size(1))

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}', 'chain')
        save_xyz_file(str(outdir), one_hot_flat, x_flat, self.lig_type_decoder,
                      name='/chain', batch_mask=mask_flat)
        visualize_chain(str(outdir), self.dataset_info, wandb=wandb)

    def prepare_pocket(self, biopython_residues, repeats=1):

        if self.pocket_representation == 'CA':
            pocket_coord = torch.tensor(np.array(
                [res['CA'].get_coord() for res in biopython_residues]),
                device=self.device, dtype=FLOAT_TYPE)
            pocket_types = torch.tensor(
                [self.pocket_type_encoder[three_to_one(res.get_resname())]
                 for res in biopython_residues], device=self.device)
        else:
            pocket_atoms = [a for res in biopython_residues
                            for a in res.get_atoms()
                            if (a.element.capitalize() in self.pocket_type_encoder or a.element != 'H')]
            pocket_coord = torch.tensor(np.array(
                [a.get_coord() for a in pocket_atoms]),
                device=self.device, dtype=FLOAT_TYPE)
            pocket_types = torch.tensor(
                [self.pocket_type_encoder[a.element.capitalize()]
                 for a in pocket_atoms], device=self.device)

        pocket_one_hot = F.one_hot(
            pocket_types, num_classes=len(self.pocket_type_encoder)
        )

        pocket_size = torch.tensor([len(pocket_coord)] * repeats,
                                   device=self.device, dtype=INT_TYPE)
        pocket_mask = torch.repeat_interleave(
            torch.arange(repeats, device=self.device, dtype=INT_TYPE),
            len(pocket_coord)
        )

        pocket = {
            'x': pocket_coord.repeat(repeats, 1),
            'one_hot': pocket_one_hot.repeat(repeats, 1),
            'size': pocket_size,
            'mask': pocket_mask
        }

        return pocket

    def generate_ligands(self, pdb_file, n_samples, pocket_ids=None,
                         ref_ligand=None, num_nodes_lig=None, sanitize=False,
                         largest_frag=False, relax_iter=0, timesteps=None,
                         n_nodes_bias=0, n_nodes_min=0, **kwargs):
        """
        Generate ligands given a pocket
        Args:
            pdb_file: PDB filename
            n_samples: number of samples
            pocket_ids: list of pocket residues in <chain>:<resi> format
            ref_ligand: alternative way of defining the pocket based on a
                reference ligand given in <chain>:<resi> format if the ligand is
                contained in the PDB file, or path to an SDF file that
                contains the ligand
            num_nodes_lig: number of ligand nodes for each sample (list of
                integers), sampled randomly if 'None'
            sanitize: whether to sanitize molecules or not
            largest_frag: only return the largest fragment
            relax_iter: number of force field optimization steps
            timesteps: number of denoising steps, use training value if None
            n_nodes_bias: added to the sampled (or provided) number of nodes
            n_nodes_min: lower bound on the number of sampled nodes
            kwargs: additional inpainting parameters
        Returns:
            list of molecules
        """

        assert (pocket_ids is None) ^ (ref_ligand is None)

        self.ddpm.eval()

        # Load PDB
        pdb_struct = PDBParser(QUIET=True).get_structure('', pdb_file)[0]
        if pocket_ids is not None:
            # define pocket with list of residues
            residues = [
                pdb_struct[x.split(':')[0]][(' ', int(x.split(':')[1]), ' ')]
                for x in pocket_ids]

        else:
            # define pocket with reference ligand
            residues = utils.get_pocket_from_ligand(pdb_struct, ref_ligand)

        pocket = self.prepare_pocket(residues, repeats=n_samples)

        # Pocket's center of mass
        pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

        # Create dummy ligands
        if num_nodes_lig is None:
            num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])

        # Add bias
        num_nodes_lig = num_nodes_lig + n_nodes_bias

        # Apply minimum ligand size
        num_nodes_lig = torch.clamp(num_nodes_lig, min=n_nodes_min)

        # Use inpainting
        if type(self.ddpm) == EnVariationalDiffusion:
            lig_mask = utils.num_nodes_to_batch_mask(
                len(num_nodes_lig), num_nodes_lig, self.device)

            ligand = {
                'x': torch.zeros((len(lig_mask), self.x_dims),
                                 device=self.device, dtype=FLOAT_TYPE),
                'one_hot': torch.zeros((len(lig_mask), self.atom_nf),
                                       device=self.device, dtype=FLOAT_TYPE),
                'size': num_nodes_lig,
                'mask': lig_mask
            }

            # Fix all pocket nodes but sample
            lig_mask_fixed = torch.zeros(len(lig_mask), device=self.device)
            pocket_mask_fixed = torch.ones(len(pocket['mask']),
                                           device=self.device)

            xh_lig, xh_pocket, lig_mask, pocket_mask = self.ddpm.inpaint(
                ligand, pocket, lig_mask_fixed, pocket_mask_fixed,
                timesteps=timesteps, **kwargs)

        # Use conditional generation
        elif type(self.ddpm) == ConditionalDDPM:
            xh_lig, xh_pocket, lig_mask, pocket_mask = \
                self.ddpm.sample_given_pocket(pocket, num_nodes_lig,
                                              timesteps=timesteps)

        else:
            raise NotImplementedError

        # Move generated molecule back to the original pocket position
        pocket_com_after = scatter_mean(
            xh_pocket[:, :self.x_dims], pocket_mask, dim=0)

        xh_pocket[:, :self.x_dims] += \
            (pocket_com_before - pocket_com_after)[pocket_mask]
        xh_lig[:, :self.x_dims] += \
            (pocket_com_before - pocket_com_after)[lig_mask]

        # Build mol objects
        x = xh_lig[:, :self.x_dims].detach().cpu()
        atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()
        lig_mask = lig_mask.cpu()

        molecules = []
        for mol_pc in zip(utils.batch_to_list(x, lig_mask),
                          utils.batch_to_list(atom_type, lig_mask)):

            mol = build_molecule(*mol_pc, self.dataset_info, add_coords=True)
            mol = process_molecule(mol,
                                   add_hydrogens=False,
                                   sanitize=sanitize,
                                   relax_iter=relax_iter,
                                   largest_frag=largest_frag)
            if mol is not None:
                molecules.append(mol)

        return molecules

    def generate_ligands_rl(self, pdb_file, n_samples, pocket_ids=None,
                         ref_ligand=None, num_nodes_lig=None, sanitize=False,
                         largest_frag=False, relax_iter=0, timesteps=None,
                         n_nodes_bias=0, n_nodes_min=0, ppo_config=None,
                         policy_update=True, optimizer=None, **kwargs):
        """
        Generate ligands given a pocket
        Args:
            pdb_file: PDB filename
            n_samples: number of samples
            pocket_ids: list of pocket residues in <chain>:<resi> format
            ref_ligand: alternative way of defining the pocket based on a
                reference ligand given in <chain>:<resi> format if the ligand is
                contained in the PDB file, or path to an SDF file that
                contains the ligand
            num_nodes_lig: number of ligand nodes for each sample (list of
                integers), sampled randomly if 'None'
            sanitize: whether to sanitize molecules or not
            largest_frag: only return the largest fragment
            relax_iter: number of force field optimization steps
            timesteps: number of denoising steps, use training value if None
            n_nodes_bias: added to the sampled (or provided) number of nodes
            n_nodes_min: lower bound on the number of sampled nodes
            kwargs: additional inpainting parameters
        Returns:
            list of molecules
        """

        assert (pocket_ids is None) ^ (ref_ligand is None)

        ppo_config.episode_length = ppo_config.max_time_steps // ppo_config.inference_interval  # length of PPO trajectory
        if policy_update and optimizer is None:
            optimizer = torch.optim.AdamW(
                self.ddpm.parameters(),
                lr=ppo_config.lr, amsgrad=True,
                weight_decay=1e-4)

        # Load PDB
        pdb_struct = PDBParser(QUIET=True).get_structure('', pdb_file)[0]
        if pocket_ids is not None:
            # define pocket with list of residues
            residues = [
                pdb_struct[x.split(':')[0]][(' ', int(x.split(':')[1]), ' ')]
                for x in pocket_ids]

        else:
            # define pocket with reference ligand
            residues = utils.get_pocket_from_ligand(pdb_struct, ref_ligand)

        pocket = self.prepare_pocket(residues, repeats=n_samples)

        # Pocket's center of mass
        pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

        # Create dummy ligands
        if num_nodes_lig is None:
            num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])

        # Add bias
        num_nodes_lig = num_nodes_lig + n_nodes_bias

        # Apply minimum ligand size
        num_nodes_lig = torch.clamp(num_nodes_lig, min=n_nodes_min)


        # Use inpainting
        if type(self.ddpm) == EnVariationalDiffusion:
            lig_mask = utils.num_nodes_to_batch_mask(
                len(num_nodes_lig), num_nodes_lig, self.device)

            ligand = {
                'x': torch.zeros((len(lig_mask), self.x_dims),
                                 device=self.device, dtype=FLOAT_TYPE),
                'one_hot': torch.zeros((len(lig_mask), self.atom_nf),
                                       device=self.device, dtype=FLOAT_TYPE),
                'size': num_nodes_lig,
                'mask': lig_mask
            }

            # Fix all pocket nodes but sample
            lig_mask_fixed = torch.zeros(len(lig_mask), device=self.device)
            pocket_mask_fixed = torch.ones(len(pocket['mask']),
                                           device=self.device)

            xh_lig, xh_pocket, lig_mask, pocket_mask = self.ddpm.inpaint(
                ligand, pocket, lig_mask_fixed, pocket_mask_fixed,
                timesteps=timesteps, **kwargs)

        # Use conditional generation
        elif type(self.ddpm) == ConditionalDDPM:
            xh_lig, xh_pocket, lig_mask, pocket_mask, rollout_buffer= \
                self.ddpm.rl_given_pocket(pocket, num_nodes_lig, return_frames=1, timesteps=None, ppo_config=ppo_config, inference_interval=ppo_config.inference_interval, optimizer=None, pocket_com_before=pocket_com_before)

        # Move generated molecule back to the original pocket position
        pocket_com_after = scatter_mean(
            xh_pocket[:, :self.x_dims], pocket_mask, dim=0)

        xh_pocket[:, :self.x_dims] += \
            (pocket_com_before - pocket_com_after)[pocket_mask]
        xh_lig[:, :self.x_dims] += \
            (pocket_com_before - pocket_com_after)[lig_mask]

        # Compute rewards and advantages if a reward function is provided
        episodic_metrics = {}
        molecules = []
        rewards = []
        sample_records = []
        if hasattr(ppo_config, "reward_fn_dict"):

            x = xh_lig[:, :self.x_dims].detach().cpu()
            atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()
            aa_type = xh_pocket[:, self.x_dims:].argmax(1).detach().cpu()
            mol_pcs = list(
                zip(
                    utils.batch_to_list(x, lig_mask.cpu()),
                    utils.batch_to_list(atom_type, lig_mask.cpu()),
                )
            )

            def _compute_reward(mol_pc, reward_fn_dict, add_hydrogens, sanitize, relax_iter, largest_frag):
                # create a reward_dict based on keys in reward_fn_dict
                reward_dict = {'valid': 0} # valid indicates check_molecule_connectivity and sanitization check
                reward_dict['connectivity'] = 0.0  # default connectivity to 0.0
                # for key in reward_fn_dict:
                #     reward_dict[key] = 0.0
                reward_dict['qed'] = 0.0  # default qed to 0.0
                reward_dict['sa'] = 0.0  # default sa to 0.0
                reward_dict['vina_score'] = 0.0  # default vina_score to 0.0
                reward_dict['vina_min'] = 0.0  # default vina_min to 0.0
                reward_dict['vina_dock'] = 0.0  #
                reward_dict['sc_rmsd'] = 0.0  #
                reward_dict['strain'] = 0.0  # default strain energy to 0.0
                reward_dict['clash'] = 0.0  # default clash to 0.0
                reward_dict['hb_donor'] = 0.0  # default hb_donor to 0.0
                reward_dict['hb_acceptor'] = 0.0  # default hb_acceptor to 0.0
                reward_dict['vdw'] = 0.0  # default vdw to 0.0
                reward_dict['hydrophobic'] = 0.0  # default hydrophobic to 0.0
                mol = build_molecule(*mol_pc, self.dataset_info, add_coords=True)

                # identify if there are isolated atoms
                if not utils.check_molecule_connectivity(mol):
                    return None, reward_dict
                n_mol_atoms = mol.GetNumAtoms()
                mol = process_molecule(
                    mol,
                    add_hydrogens=add_hydrogens,
                    sanitize=sanitize,
                    relax_iter=relax_iter,
                    largest_frag=largest_frag,
                )
                if mol is None: # Sanitization failed
                    return None, reward_dict

                if largest_frag:
                    if mol.GetNumAtoms() >= n_mol_atoms:
                        reward_dict['connectivity'] = 1.0
                # check connectivity threshold when largest_frag is false
                if not largest_frag:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    if largest_mol.GetNumAtoms() / n_mol_atoms < self.ligand_metrics.connectivity_thresh:
                        return None, reward_dict
                    else:
                        reward_dict['connectivity'] = 1.0

                  # to ensure molecule is processed
                # r = ppo_config.reward_fn(mol)
                if 'vina_score' in reward_fn_dict:
                    vina_score = reward_fn_dict['vina_score'](mol)
                    if np.isnan(vina_score):
                        return None, reward_dict
                    reward_dict['vina_score'] = -vina_score
                if 'vina_min' in reward_fn_dict:
                    vina_min = reward_fn_dict['vina_min'](mol)
                    if np.isnan(vina_min):
                        return None, reward_dict
                    reward_dict['vina_min'] = -vina_min
                    # reward_dict['sc_rmsd'] = vina_min_dict['symmetry_rmsd']
                if 'vina_dock' in reward_fn_dict:
                    vina_dock = reward_fn_dict['vina_dock'](mol)
                    if np.isnan(vina_dock):
                        return None, reward_dict
                    reward_dict['vina_dock'] = -vina_dock
                    # reward_dict['sc_rmsd'] = vina_dock_dict['symmetry_rmsd']
                if 'qed' in reward_fn_dict:
                    reward_dict['qed'] = reward_fn_dict['qed'](mol)
                if 'sa' in reward_fn_dict:
                    reward_dict['sa'] = reward_fn_dict['sa'](mol)
                if 'posecheck' in reward_fn_dict:
                    pose_check_results = reward_fn_dict['posecheck'](mol)
                    if pose_check_results.get('strain'):
                        reward_dict['strain'] = -pose_check_results['strain']
                    if pose_check_results.get('clash'):
                        reward_dict['clash'] = -pose_check_results['clash']
                    # reward_dict.update(pose_check_results)

                # if isinstance(r, (tuple, list)):
                #     r = r[0]
                # if r is None or not np.isfinite(r):
                reward_dict['valid'] = 1
                return mol, reward_dict

            def _compute_fps_distances(results):
                mol_list = [mol for mol, _ in results] 
                fps = [Chem.RDKFingerprint(m) if m is not None else None for m in mol_list]
                div_raw = np.zeros(len(mol_list), np.float32)
                for i, fpi in enumerate(fps):
                    if fpi is None:
                        continue
                    sims = [
                        DataStructs.TanimotoSimilarity(fpi, fpj)
                        for j, fpj in enumerate(fps)
                        if j != i and fpj is not None
                    ]
                    if sims:
                        div_raw[i] = 1.0 - np.mean(sims)  # “how different mol i is from others”
                return div_raw

            def _postprocess(array, valid_array, mode='rank'):
                if mode == 'rank':
                    norm_array = utils.rank01_masked(array, valid_array, rescale=False)
                    norm_array = utils.pct_to_normal(norm_array)
                elif mode == 'minmax':
                    norm_array = utils.minmax_normalize_masked(array, valid_array)
                elif mode == 'zscore':
                    norm_array = utils.zscore_normalize_masked(array, valid_array)
                return np.nan_to_num(norm_array,  nan=0.0)

            max_workers = getattr(ppo_config, "reward_num_workers", None)
            if max_workers is None:
                # Default to 1 to avoid collisions for reward fns that write temp files
                max_workers = 3
            reward_worker = partial(
                _compute_reward,
                reward_fn_dict=ppo_config.reward_fn_dict,
                add_hydrogens=False,            # or expose as a function argument
                sanitize=sanitize,
                relax_iter=relax_iter,
                largest_frag=largest_frag,
            )
            with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(reward_worker, mol_pcs))

            analyze_out = self.analyze_sample(mol_pcs, atom_type.tolist(), aa_type.tolist(), receptors=None)
            # rewards_list = []
            properties_dict = defaultdict(list)
            for mol, reward_dict in results:
                if mol is not None:
                    molecules.append(mol)
                # rewards_list.append(reward_dict['rew'])  # Example: add diversity bonus
                for key, value in reward_dict.items():
                    properties_dict[key].append(value)
            valid_array = np.array(properties_dict['valid'], dtype=bool)
            distance_array = _compute_fps_distances(results) # compute diversity based on fingerprint distances
            mean_distance = np.mean(distance_array[valid_array]) if np.sum(valid_array) > 0 else 0.0
            norm_distance = _postprocess(distance_array, valid_array)

            # -------------------eval posecheck for all molecules-------------------
            # pose_check_results = ppo_config.reward_fn_dict['posecheck'](molecules)
            # # length of molecules is smaller than that of results,
            # # take properties_dict['valid'] into consideration，insert 0.0 for invalid molecules
            # updated_pose_check_results = defaultdict(list)
            # mol_idx = 0
            # for is_valid in properties_dict['valid']:
            #     if is_valid:
            #         for key, value in pose_check_results.items():
            #             updated_pose_check_results[key].append(value[mol_idx])
            #         mol_idx += 1
            #     else:
            #         for key in pose_check_results.keys():
            #             updated_pose_check_results[key].append(0.0)
            # pose_check_results = updated_pose_check_results
            # ----------------------------------------------------------------------

            if 'qed' in properties_dict:
                qed_array = np.array(properties_dict['qed'])
                norm_qed = _postprocess(qed_array, valid_array)
                # norm_qed = utils.rank01_masked(qed_array, valid_array, rescale=False)
                # norm_qed = utils.minmax_normalize_masked(qed_array, valid_array)
                # norm_qed = utils.pct_to_normal(norm_qed)
                # norm_qed = np.nan_to_num(norm_qed,  nan=0.0)
            if 'sa' in properties_dict:
                sa_array = np.array(properties_dict['sa'])
                norm_sa = _postprocess(sa_array, valid_array)
                # norm_sa = utils.rank01_masked(sa_array, valid_array, rescale=False)
                # norm_sa = utils.minmax_normalize_masked(sa_array, valid_array)
                # norm_sa = utils.pct_to_normal(norm_sa)
                # norm_sa = np.nan_to_num(norm_sa,  nan=0.0)
            if 'vina_score' in properties_dict:
                vina_score_array = np.array(properties_dict['vina_score'])
                norm_vina_score = _postprocess(vina_score_array, valid_array)
                # norm_vina_score = utils.rank01_masked(vina_score_array, valid_array, rescale=False)
                # norm_vina_score = utils.minmax_normalize_masked(vina_score_array, valid_array)
                # norm_vina_score = utils.pct_to_normal(norm_vina_score)
                # norm_vina_score = np.nan_to_num(norm_vina_score,  nan=0.0)
            if 'vina_dock' in properties_dict:
                vina_dock_array = np.array(properties_dict['vina_dock'])
                norm_vina_dock = _postprocess(vina_dock_array, valid_array)
                # norm_vina_dock = utils.rank01_masked(vina_dock_array, valid_array, rescale=False)
                # norm_vina_dock = utils.minmax_normalize_masked(vina_dock_array, valid_array)
                # norm_vina_dock = utils.pct_to_normal(norm_vina_dock)
                # norm_vina_dock = np.nan_to_num(norm_vina_dock,  nan=0.0)
            if 'strain' in properties_dict:
                strain_array = np.array(properties_dict['strain'])
                norm_strain = _postprocess(strain_array, valid_array)
            if 'clash' in properties_dict:
                clash_array = np.array(properties_dict['clash'])
                norm_clash = _postprocess(clash_array, valid_array)
            # combined = norm_qed * 0.27 + norm_sa * 0.13 + norm_vina_score * 0.3 + norm_distance * 0.3
            # combined = norm_qed * 0.3 + norm_sa * 0.15 + norm_vina_score * 0.3 + norm_distance * 0.25 # minmax
            # combined = norm_qed * 0.3 + norm_sa * 0.3 + norm_vina_score * 0.4 + norm_distance * 0.0 # with kl coeff
            w_qed = ppo_config.reward_weights['qed']
            w_sa = ppo_config.reward_weights['sa']
            w_vina_score = ppo_config.reward_weights['vina_score']
            w_distance = ppo_config.reward_weights['distance']
            w_strain = ppo_config.reward_weights.get('strain', 0.0)
            combined = norm_qed * w_qed + norm_sa * w_sa + norm_vina_score * w_vina_score + norm_distance * w_distance
            if w_strain > 0.0:
                combined += norm_strain * w_strain
            # -------A more flexible way to handle multiple properties--------
            # property_arrays = {}
            # norm_properties = {}
            # for prop_name, values in properties_dict.items():
            #     if prop_name == 'valid':
            #         continue
            #     prop_array = np.array(values)
            #     property_arrays[prop_name] = prop_array
            #     norm_properties[prop_name] = _postprocess(prop_array, valid_array, mode='rank')

            # def _get_or_zeros(source_dict, key):
            #     value = source_dict.get(key)
            #     if value is None:
            #         return np.zeros(len(valid_array), dtype=np.float32)
            #     return value

            # # norm_qed = _get_or_zeros(norm_properties, 'qed')
            # # norm_sa = _get_or_zeros(norm_properties, 'sa')
            # # norm_vina_score = _get_or_zeros(norm_properties, 'vina_score')
            # # norm_vina_dock = _get_or_zeros(norm_properties, 'vina_dock')

            # combined = np.zeros(len(valid_array), dtype=np.float32)
            # if norm_properties.get('qed') is not None:
            #     qed_weight = ppo_config.property_weights['qed']
            #     combined += norm_properties['qed'] * qed_weight
            # if norm_properties.get('sa') is not None:
            #     sa_weight = ppo_config.property_weights['sa']
            #     combined += norm_properties['sa'] * sa_weight
            # if norm_properties.get('vina_score') is not None:
            #     vina_score_weight = ppo_config.property_weights['vina_score']
            #     combined += norm_properties['vina_score'] * vina_score_weight
            # if norm_properties.get('vina_dock') is not None:
            #     vina_dock_weight = ppo_config.property_weights['vina_dock']
            #     combined += norm_properties['vina_dock'] * vina_dock_weight
            # if ppo_config.property_weights['diversity'] is not None:
            #     diversity_weight = ppo_config.property_weights['diversity']
            #     combined += norm_distance * diversity_weight
            # ---------------------------------------------------------------


            # qed_array[valid_array] = np.minimum(qed_array[valid_array], 0.6)  # cap max qed to avoid dominating the reward
            # sa_array[valid_array] = np.minimum(sa_array[valid_array], 0.9)  # cap max sa to avoid dominating the reward
            # distance_array[valid_array] = np.minimum(distance_array[valid_array], 0.9)  # cap max distance to avoid dominating the reward

            # list of penalty for sa if lower than 0.5, else 0, leave invalids ignored
            # sa_penalty_array = np.array(sa_array)  # make a copy
            # valid_mask = valid_array == 1
            # low_sa_mask = (sa_array < 0.7) & valid_mask
            # sa_penalty_array[valid_mask] = 0.0
            # sa_penalty_array[low_sa_mask] = sa_array[low_sa_mask] - 0.7

            # Compute ranks for each property, range (0,1)

            
            # norm_qed = utils.rank01_masked(qed_array, valid_array, rescale=False)
            # norm_sa = utils.rank01_masked(sa_array, valid_array, rescale=False)
            # norm_sa = utils.rank01_masked(list_sa_penalty, list_valid, rescale=False)
            # norm_vina_score = utils.rank01_masked(vina_score_array, valid_array, rescale=False)
            # norm_vina_dock = utils.rank01_masked(vina_dock_array, valid_array, rescale=False)
            # norm_distance = utils.rank01_masked(distance_array, valid_array, rescale=False)
            # norm_qed = utils.percentile_rank_masked(list_qed, list_valid, rescale=False)
            # # norm_sa = utils.percentile_rank_masked(list_sa, list_valid, rescale=False)
            # norm_sa = utils.percentile_rank_masked(list_sa_penalty, list_valid, rescale=False)
            # norm_docking = utils.percentile_rank_masked(list_docking, list_valid, rescale=False)

            # minmax normalize to (0,1) 
            # norm_qed = utils.minmax_normalize_masked(qed_array, valid_array)
            # norm_sa = utils.minmax_normalize_masked(sa_array, valid_array)
            # norm_docking = utils.minmax_normalize_masked(docking_array, valid_array)
            # norm_distance = utils.minmax_normalize_masked(distance_array, valid_array)

            # Map to N(0,1); “rank-to-normal” transform or “quantile normalization.”
            # norm_qed = utils.pct_to_normal(norm_qed)
            # norm_sa = utils.pct_to_normal(norm_sa)
            # norm_vina_score = utils.pct_to_normal(norm_vina_score)
            # norm_vina_dock = utils.pct_to_normal(norm_vina_dock)
            # norm_distance = utils.pct_to_normal(norm_distance)

            # norm_qed = _postprocess(qed_array, valid_array, mode='rank') if qed_array is not None else None
            # norm_sa = _postprocess(sa_array, valid_array, mode='rank') if sa_array is not None else None
            # norm_vina_score = _postprocess(vina_score_array, valid_array, mode='rank') if vina_score_array is not None else None
            # norm_vina_dock = _postprocess(vina_dock_array, valid_array, mode='rank') if vina_dock_array is not None else None
            # norm_distance = _postprocess(distance_array, valid_array, mode='rank') if distance_array is not None else None

            # Combine valid scores
            # norm_qed = np.nan_to_num(norm_qed,  nan=0.0) if norm_qed is not None else None
            # norm_sa = np.nan_to_num(norm_sa,   nan=0.0) if norm_sa is not None else None
            # norm_vina_score = np.nan_to_num(norm_vina_score, nan=0.0) if norm_vina_score is not None else None
            # norm_vina_dock = np.nan_to_num(norm_vina_dock, nan=0.0) if norm_vina_dock is not None else None
            # norm_distance = np.nan_to_num(norm_distance, nan=0.0) if norm_distance is not None else None
            # check if norm_qed is not all-zero array

            # combined = norm_qed * 0.3 + norm_sa * 0.1 + norm_vina_score * 0.3 + norm_distance * 0.3
            # combined = norm_qed * 0.28 + norm_sa * 0.17 + norm_vina_score * 0.3 + norm_distance * 0.25 
            # if norm_gap_vina_score_dock is not None:
            #     combined += (-1.0) * norm_gap_vina_score_dock * 0.1  # penalize large gap between vina score and docking score
            # valid_idx = np.where(valid_array)[0]
            # if valid_idx.size > 0:
            #     norm_qed = np.nan_to_num(norm_qed,  nan=0.0)
            #     norm_sa = np.nan_to_num(norm_sa,   nan=0.0)
            #     norm_vina_score = np.nan_to_num(norm_vina_score, nan=0.0)
            #     norm_vina_dock = np.nan_to_num(norm_vina_dock, nan=0.0)
            #     norm_distance = np.nan_to_num(norm_distance, nan=0.0)
            #     combined = norm_qed * 0.3 + norm_sa * 0.1 + norm_vina_score * 0.35 + norm_distance * 0.25 
                # combined = norm_qed * 0.33 + norm_sa * 0.1 + norm_vina_score * 0.35 + norm_distance * 0.22 
                # combined = qed_array * 0.3 + sa_array * 0.1 + norm_vina_score * 0.3 + distance_array * 0.3
            
            # Assign fixed penalty to invalids
            FIXED_INVALID_PENALTY = -3.09 # norm.ppf(1e-3) ≈ -3.09
            # FIXED_INVALID_PENALTY = -1.0 + 1e-4

            # handle invalid samples by assigning them a score of (mean - 5*std)
            valid_scores = combined[valid_array]
            mu, sigma = valid_scores.mean(), valid_scores.std() + 1e-8
            combined[~valid_array] = mu - 3.0 * sigma  # assign invalids to be 3 std below mean
            # combined[~valid_array] = FIXED_INVALID_PENALTY

            # combined = np.array(list_qed) * 0.8 + np.array(list_sa) * 0.2 + np.array(list_docking) * 0.1
            # combined[~valid_array] = -0.1

            # Prepare sample records
            for idx, (mol, reward_dict) in enumerate(results):
                if mol is None:
                    continue
                sample_records.append({
                    "mol": mol,
                    "metrics": {
                        "qed": float(reward_dict.get("qed", 0.0)),
                        "sa": float(reward_dict.get("sa", 0.0)),
                        "vina_score": float(reward_dict.get("vina_score", 0.0)),
                        "vina_min": float(reward_dict.get("vina_min", 0.0)),
                        "vina_dock": float(reward_dict.get("vina_dock", 0.0)),
                        "sc_rmsd": float(reward_dict.get("sc_rmsd", 0.0)),
                        "strain": float(reward_dict.get("strain", 0.0)),
                        "clash": float(reward_dict.get("clash", 0.0)),
                        "distance": float(distance_array[idx]),
                        "num_atoms": int(mol.GetNumAtoms()),
                        "connectivity": float(reward_dict.get("connectivity", 0.0)),
                    }
                })

            rewards = torch.tensor(combined, device=self.device, dtype=torch.float32)
            rewards_mean = rewards.mean()
            rewards_std = rewards.std()
            advantages = (rewards - rewards_mean) / (rewards_std + 1e-8)

            def _get_mean_from_records(records, key):
                if len(records) == 0:
                    return 0.0
                return np.mean([rec['metrics'][key] for rec in records])

            # Log episodic metrics for logging, like wandb
            episodic_metrics["combined_rewards"] = rewards_mean.item()
            episodic_metrics["combined_reward_std"] = rewards_std.item()
            episodic_metrics["valid_combined_mu"] = mu
            episodic_metrics["valid_combined_std"] = sigma
            episodic_metrics["qed"] = _get_mean_from_records(sample_records, 'qed')
            episodic_metrics["sa"] = _get_mean_from_records(sample_records, 'sa')
            episodic_metrics["docking_score"] = _get_mean_from_records(sample_records, 'vina_score')
            episodic_metrics["vina_score"] = _get_mean_from_records(sample_records, 'vina_score')
            episodic_metrics["vina_min"] = _get_mean_from_records(sample_records, 'vina_min')
            episodic_metrics["vina_dock"] = _get_mean_from_records(sample_records, 'vina_dock')
            episodic_metrics["strain"] = _get_mean_from_records(sample_records, 'strain')
            episodic_metrics["clash"] = _get_mean_from_records(sample_records, 'clash')
            episodic_metrics["connectivity"] = _get_mean_from_records(sample_records, 'connectivity')
            episodic_metrics["distance"] = mean_distance
            episodic_metrics["valid_rate"] = float(len(molecules)) / float(len(results)) if len(results) > 0 else 0.0
            episodic_metrics["mean_size"] = sum([mol.GetNumAtoms() for mol in molecules]) / len(molecules) if len(molecules) > 0 else 0.0
            # episodic_metrics["qed"] = analyze_out['QED']
            # episodic_metrics["sa"] = analyze_out['SA']
            # episodic_metrics["logp"] = analyze_out['LogP']
            episodic_metrics['diversity'] = analyze_out['Diversity']
            episodic_metrics["Diversity"] = analyze_out['Diversity']
            episodic_metrics["Validity"] = analyze_out['Validity']
            episodic_metrics["Connectivity"] = analyze_out['Connectivity']
            episodic_metrics["QED"] = analyze_out['QED']
            episodic_metrics["SA"] = analyze_out['SA']
            episodic_metrics["LogP"] = analyze_out['LogP']
            episodic_metrics["Lipinski"] = analyze_out['Lipinski']
        else:
            # No reward function provided; zero-advantages prevents updates
            print("No reward function provided in PPO config; skipping reward computation.")
            rewards = torch.zeros(n_samples, device=self.device)
            advantages = torch.zeros(n_samples, device=self.device)
        
        # print(f"Rollout ligand sizes: {num_nodes_lig}")
        # print(f"Rollout rewards: {rewards_list}")p

        # temp_metric = MoleculeProperties()
        # pocket_rdmols = []
        # if len(molecules) > 0:
        #     (validity, connectivity, uniqueness, novelty), (_, connected_mols) = self.ligand_metrics.evaluate_rdmols(molecules)

        #     qed, sa, logp, lipinski, diversity = self.molecule_properties.evaluate_mean(connected_mols)
        
        # dict_ligsize_reward = {str(sz): rw for sz, rw in zip(num_nodes_lig.tolist(), rewards_list)}
        # print(f"Rollout ligand sizes and rewards: {dict_ligsize_reward} ")
        print(f"valid_array: {np.array(properties_dict['valid'])}")
        print(f"vina_score: {properties_dict['vina_score']}")
        print(f"vina_min: {properties_dict['vina_min']}")
        print(f"vina_dock: {properties_dict['vina_dock']}")
        print(f"strain energy: {properties_dict['strain']}")
        print(f"clash score: {properties_dict['clash']}")
        print("pdb_file:", pdb_file)
        print("Episodic metrics:", episodic_metrics)
        print(f"Rollout combined rewards mean: {rewards.mean().item()}, std: {rewards.std().item()}")

        if policy_update:
            # PPO update
            # Time indices
            time_indices = np.arange(ppo_config.episode_length)

            # Pre-normalize stored times
            s_norm_all = rollout_buffer.s_steps / ppo_config.max_time_steps
            t_norm_all = rollout_buffer.t_steps / ppo_config.max_time_steps

            np.random.shuffle(time_indices)

            approx_kl_vals = []
            clipfrac_vals = []
            loss_vals = []

            for start in range(0, ppo_config.episode_length, ppo_config.batch_size):
                end = min(start + ppo_config.batch_size, ppo_config.episode_length)
                indices = time_indices[start:end]

                for i in indices:
                    # Recompute log_prob under current policy for PPO ratio
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out_dict = self.ddpm.sample_p_zs_given_zt_rl(
                            s_norm_all[i], t_norm_all[i],
                            zt_lig=rollout_buffer.obs_steps[i], xh0_pocket=rollout_buffer.xh_pocket_steps[i],
                            ligand_mask=lig_mask, pocket_mask=pocket['mask'],
                            action=rollout_buffer.action_steps[i], mu_via_x0=True
                        )
                        new_log_prob = out_dict['log_prob']        # (n_samples,)
                        old_log_prob = rollout_buffer.log_prob_steps[i] 
                        # is num_nodes_lig contains zeros, then corresponding log_prob would be nan, need to filter them out
                        new_log_prob = new_log_prob[~new_log_prob.isnan()]
                        old_log_prob = old_log_prob[~old_log_prob.isnan()]

                        logratio = new_log_prob - old_log_prob
                        ratio = torch.exp(logratio)

                        # Optionally clip advantages if desired (not provided here)
                        unclipped = -advantages * ratio
                        clipped = -advantages * torch.clamp(
                            ratio,
                            1.0 - ppo_config.clip_range,
                            1.0 + ppo_config.clip_range,
                        )
                        total_loss = 0.0
                        loss_i = torch.mean(torch.maximum(unclipped, clipped))
                        total_loss = total_loss + loss_i
                        # Diagnostics similar to ddpo
                        approx_kl_vals.append(0.5 * torch.mean(logratio ** 2).detach().item())
                        clipfrac_vals.append(torch.mean((torch.abs(ratio - 1.0) > ppo_config.clip_range).float()).detach().item())
                        loss_vals.append(loss_i.detach().item())
                        # --- KL to pretrained policy ---
                        if hasattr(self, "ddpm_pretrained") and ppo_config.kl_coeff_pretrain > 0:
                            with torch.no_grad():
                            #     # Also add KL penalty to the pretrained model to prevent
                            #     # catastrophic policy updates
                            #     # Compute log_prob under the pretrained model
                                pretrained_out_dict = self.ddpm_pretrained.sample_p_zs_given_zt_rl(
                                    s_norm_all[i], t_norm_all[i],
                                    zt_lig=rollout_buffer.obs_steps[i], xh0_pocket=rollout_buffer.xh_pocket_steps[i],
                                    ligand_mask=lig_mask, pocket_mask=pocket['mask'],
                                    action=rollout_buffer.action_steps[i], mu_via_x0=True
                                )
                                pretrained_log_prob = pretrained_out_dict['log_prob']
                            pretrained_log_prob = pretrained_log_prob[~pretrained_log_prob.isnan()]
                            new_log_prob = new_log_prob[~new_log_prob.isnan()]
                            # # kl divergence to the pretrained model 
                            logratio_ref_theta = pretrained_log_prob - new_log_prob
                            ratio_ref_theta = torch.exp(logratio_ref_theta)
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            approx_kl = (ratio_ref_theta - 1 - logratio_ref_theta)

                            # without importance sampling weights
                            kl_loss = approx_kl.mean()
                            # with importance sampling weights
                            importance_sampling_weights = ratio
                            # kl_loss = (importance_sampling_weights * approx_kl).mean()
                            
                            # kl_loss = torch.mean(new_log_prob - pretrained_log_prob)
                            total_loss = total_loss + ppo_config.kl_coeff_pretrain * kl_loss
                    # -------------------------------
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ddpm.dynamics.parameters(),
                                                    max_norm=1.0)
                    optimizer.step()

                    # [when using scaler = torch.amp.GradScaler()]
                    # optimizer.zero_grad()
                    # scaler.scale(total_loss).backward()
                    # scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(self.ddpm.dynamics.parameters(),
                    #                                 max_norm=1.0)
                    # scaler.step(optimizer)
                    # scaler.update()

            # metrics = {"rewards": rewards.mean().item()}

            episodic_metrics.update({
                "approx_kl": float(np.mean(approx_kl_vals)),
                "clipfrac": float(np.mean(clipfrac_vals)),
                "loss": float(np.mean(loss_vals)),
            })

        return episodic_metrics, sample_records

    def configure_gradient_clipping(self, optimizer,
                                    gradient_clip_val, gradient_clip_algorithm):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + \
                        2 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')


class WeightSchedule:
    def __init__(self, T, max_weight, mode='linear'):
        if mode == 'linear':
            self.weights = torch.linspace(max_weight, 0, T + 1)
        elif mode == 'constant':
            self.weights = max_weight * torch.ones(T + 1)
        else:
            raise NotImplementedError(f'{mode} weight schedule is not '
                                      f'available.')

    def __call__(self, t_array):
        """ all values in t_array are assumed to be integers in [0, T] """
        return self.weights[t_array].to(t_array.device)
