import math

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

import utils
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from equivariant_diffusion.s_predictor import sample_time_interval_scale
from types import SimpleNamespace

from equivariant_diffusion.s_predictor import SPredictor, EGNNSPredictor


class ConditionalDDPM(EnVariationalDiffusion):
    """
    Conditional Diffusion Module.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.dynamics.update_pocket_coords

    def kl_prior(self, xh_lig, mask_lig, num_nodes):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice
        negligible in the loss. However, you compute it so that you see it when
        you've made a mistake in your noise schedule.
        """
        batch_size = len(num_nodes)

        # Compute the last alpha value, alpha_T.
        ones = torch.ones((batch_size, 1), device=xh_lig.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh_lig)

        # Compute means.
        mu_T_lig = alpha_T[mask_lig] * xh_lig
        mu_T_lig_x, mu_T_lig_h = \
            mu_T_lig[:, :self.n_dims], mu_T_lig[:, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_lig_x).squeeze()
        sigma_T_h = self.sigma(gamma_T, mu_T_lig_h).squeeze()

        # Compute KL for h-part.
        zeros = torch.zeros_like(mu_T_lig_h)
        ones = torch.ones_like(sigma_T_h)
        mu_norm2 = self.sum_except_batch((mu_T_lig_h - zeros) ** 2, mask_lig)
        kl_distance_h = self.gaussian_KL(mu_norm2, sigma_T_h, ones, d=1)

        # Compute KL for x-part.
        zeros = torch.zeros_like(mu_T_lig_x)
        ones = torch.ones_like(sigma_T_x)
        mu_norm2 = self.sum_except_batch((mu_T_lig_x - zeros) ** 2, mask_lig)
        subspace_d = self.subspace_dimensionality(num_nodes)
        kl_distance_x = self.gaussian_KL(mu_norm2, sigma_T_x, ones, subspace_d)

        return kl_distance_x + kl_distance_h

    def log_pxh_given_z0_without_constants(self, ligand, z_0_lig, eps_lig,
                                           net_out_lig, gamma_0, epsilon=1e-10):

        # Discrete properties are predicted directly from z_t.
        z_h_lig = z_0_lig[:, self.n_dims:]

        # Take only part over x.
        eps_lig_x = eps_lig[:, :self.n_dims]
        net_lig_x = net_out_lig[:, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_0_lig)
        sigma_0_cat = sigma_0 * self.norm_values[1]

        # Computes the error for the distribution
        # N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        squared_error = (eps_lig_x - net_lig_x) ** 2
        if self.vnode_idx is not None:
            # coordinates of virtual atoms should not contribute to the error
            squared_error[ligand['one_hot'][:, self.vnode_idx].bool(), :self.n_dims] = 0
        log_p_x_given_z0_without_constants_ligand = -0.5 * (
            self.sum_except_batch(squared_error, ligand['mask'])
        )

        # Compute delta indicator masks.
        # un-normalize
        ligand_onehot = ligand['one_hot'] * self.norm_values[1] + self.norm_biases[1]

        estimated_ligand_onehot = z_h_lig * self.norm_values[1] + self.norm_biases[1]

        # Centered h_cat around 1, since onehot encoded.
        centered_ligand_onehot = estimated_ligand_onehot - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional_ligand = torch.log(
            self.cdf_standard_gaussian((centered_ligand_onehot + 0.5) / sigma_0_cat[ligand['mask']])
            - self.cdf_standard_gaussian((centered_ligand_onehot - 0.5) / sigma_0_cat[ligand['mask']])
            + epsilon
        )

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional_ligand, dim=1,
                                keepdim=True)
        log_probabilities_ligand = log_ph_cat_proportional_ligand - log_Z

        # Select the log_prob of the current category using the onehot
        # representation.
        log_ph_given_z0_ligand = self.sum_except_batch(
            log_probabilities_ligand * ligand_onehot, ligand['mask'])

        return log_p_x_given_z0_without_constants_ligand, log_ph_given_z0_ligand

    def sample_p_xh_given_z0(self, z0_lig, xh0_pocket, lig_mask, pocket_mask,
                             batch_size, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        t_zeros = torch.zeros(size=(batch_size, 1), device=z0_lig.device)
        gamma_0 = self.gamma(t_zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0)
        net_out_lig, _ = self.dynamics(
            z0_lig, xh0_pocket, t_zeros, lig_mask, pocket_mask)

        # Compute mu for p(zs | zt).
        mu_x_lig = self.compute_x_pred(net_out_lig, z0_lig, gamma_0, lig_mask)
        xh_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_x_lig, xh0_pocket, sigma_x, lig_mask, pocket_mask, fix_noise)

        x_lig, h_lig = self.unnormalize(
            xh_lig[:, :self.n_dims], z0_lig[:, self.n_dims:])
        x_pocket, h_pocket = self.unnormalize(
            xh0_pocket[:, :self.n_dims], xh0_pocket[:, self.n_dims:])

        h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)
        # h_pocket = F.one_hot(torch.argmax(h_pocket, dim=1), self.residue_nf)

        return x_lig, h_lig, x_pocket, h_pocket

    def sample_p_xh_given_zt(self, zt_lig, xh0_pocket, lig_mask, pocket_mask,
                             t, fix_noise=False):
        """Samples x ~ p(x|z_t) analogous to sample_p_xh_given_z0.

        Args:
            zt_lig: (N_lig_total, n_dims + atom_nf) latent at time t
            xh0_pocket: (N_pocket_total, n_dims + residue_nf) pocket features
            lig_mask: (N_lig_total,) batch index per ligand atom
            pocket_mask: (N_pocket_total,) batch index per pocket residue
            t: (batch, 1) normalized timestep in [0, 1]
            fix_noise: if True, fix sampling noise (not implemented)

        Returns:
            x_lig, h_lig_onehot, x_pocket, h_pocket (all unnormalized)
        """
        # Compute gamma and corresponding sigma_x = sqrt(sigma_t^2 / alpha_t^2)
        gamma_t = self.gamma(t)
        sigma_x = self.SNR(-0.5 * gamma_t)

        # Network prediction at time t
        net_out_lig, _ = self.dynamics(
            zt_lig, xh0_pocket, t, lig_mask, pocket_mask)

        # Mean for p(x|z_t) via chosen parametrization
        mu_x_lig = self.compute_x_pred(net_out_lig, zt_lig, gamma_t, lig_mask)

        # Clamp only h-part (categorical logits) to [0,1] to avoid runaway
        mu_x_lig = torch.cat(
            [mu_x_lig[:, :self.n_dims],
                mu_x_lig[:, self.n_dims:].clamp(0.0, 1.0)],
            dim=1
        )

        # Sample around the mean with zero-COM projection together with pocket
        xh_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_x_lig, xh0_pocket, sigma_x, lig_mask, pocket_mask, fix_noise)

        # Unnormalize; for categorical part use z_t logits (as in z0 case)
        x_lig, h_lig = self.unnormalize(
            xh_lig[:, :self.n_dims], zt_lig[:, self.n_dims:])
        x_pocket, h_pocket = self.unnormalize(
            xh0_pocket[:, :self.n_dims], xh0_pocket[:, self.n_dims:])

        # Discretize ligand atom types to one-hot
        h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)
        # h_pocket = F.one_hot(torch.argmax(h_pocket, dim=1), self.residue_nf)

        return x_lig, h_lig, x_pocket, h_pocket

    def sample_normal(self, *args):
        raise NotImplementedError("Has been replaced by sample_normal_zero_com()")

    def sample_normal_zero_com(self, mu_lig, xh0_pocket, sigma, lig_mask,
                               pocket_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        if fix_noise:
            # bs = 1 if fix_noise else mu.size(0)
            raise NotImplementedError("fix_noise option isn't implemented yet")

        eps_lig = self.sample_gaussian(
            size=(len(lig_mask), self.n_dims + self.atom_nf),
            device=lig_mask.device)

        out_lig = mu_lig + sigma[lig_mask] * eps_lig

        # project to COM-free subspace
        xh_pocket = xh0_pocket.detach().clone()
        out_lig[:, :self.n_dims], xh_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(out_lig[:, :self.n_dims],
                                   xh0_pocket[:, :self.n_dims],
                                   lig_mask, pocket_mask)

        return out_lig, xh_pocket

    def noised_representation(self, xh_lig, xh0_pocket, lig_mask, pocket_mask,
                              gamma_t):
        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, xh_lig)
        sigma_t = self.sigma(gamma_t, xh_lig)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps_lig = self.sample_gaussian(
            size=(len(lig_mask), self.n_dims + self.atom_nf),
            device=lig_mask.device)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t_lig = alpha_t[lig_mask] * xh_lig + sigma_t[lig_mask] * eps_lig

        # project to COM-free subspace
        xh_pocket = xh0_pocket.detach().clone()
        z_t_lig[:, :self.n_dims], xh_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(z_t_lig[:, :self.n_dims],
                                   xh_pocket[:, :self.n_dims],
                                   lig_mask, pocket_mask)

        return z_t_lig, xh_pocket, eps_lig

    def log_pN(self, N_lig, N_pocket):
        """
        Prior on the sample size for computing
        log p(x,h,N) = log p(x,h|N) + log p(N), where log p(x,h|N) is the
        model's output
        Args:
            N: array of sample sizes
        Returns:
            log p(N)
        """
        log_pN = self.size_distribution.log_prob_n1_given_n2(N_lig, N_pocket)
        return log_pN

    def delta_log_px(self, num_nodes):
        return -self.subspace_dimensionality(num_nodes) * \
               np.log(self.norm_values[0])

    def forward(self, ligand, pocket, return_info=False):
        """
        Computes the loss and NLL terms
        """
        # Normalize data, take into account volume change in x.
        ligand, pocket = self.normalize(ligand, pocket)

        # Likelihood change due to normalization
        # if self.vnode_idx is not None:
        #     delta_log_px = self.delta_log_px(ligand['size'] - ligand['num_virtual_atoms'] + pocket['size'])
        # else:
        delta_log_px = self.delta_log_px(ligand['size'])

        # Sample a timestep t for each example in batch
        # At evaluation time, loss_0 will be computed separately to decrease
        # variance in the estimator (costs two forward passes)
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(ligand['size'].size(0), 1),
            device=ligand['x'].device).float()
        s_int = t_int - 1  # previous timestep

        # Masks: important to compute log p(x | z0).
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), ligand['x'])
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand['x'])

        # Concatenate x, and h[categorical].
        xh0_lig = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        # Center the input nodes
        xh0_lig[:, :self.n_dims], xh0_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(xh0_lig[:, :self.n_dims],
                                   xh0_pocket[:, :self.n_dims],
                                   ligand['mask'], pocket['mask'])

        # Find noised representation
        z_t_lig, xh_pocket, eps_t_lig = \
            self.noised_representation(xh0_lig, xh0_pocket, ligand['mask'],
                                       pocket['mask'], gamma_t)

        # Neural net prediction.
        net_out_lig, _ = self.dynamics(
            z_t_lig, xh_pocket, t, ligand['mask'], pocket['mask'])

        # For LJ loss term
        # xh_lig_hat does not need to be zero-centered as it is only used for
        # computing relative distances
        xh_lig_hat = self.xh_given_zt_and_epsilon(z_t_lig, net_out_lig, gamma_t,
                                                  ligand['mask'])

        # Compute the L2 error.
        squared_error = (eps_t_lig - net_out_lig) ** 2
        if self.vnode_idx is not None:
            # coordinates of virtual atoms should not contribute to the error
            squared_error[ligand['one_hot'][:, self.vnode_idx].bool(), :self.n_dims] = 0
        error_t_lig = self.sum_except_batch(squared_error, ligand['mask'])

        # Compute weighting with SNR: (1 - SNR(s-t)) for epsilon parametrization
        SNR_weight = (1 - self.SNR(gamma_s - gamma_t)).squeeze(1)
        assert error_t_lig.size() == SNR_weight.size()

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(
            n_nodes=ligand['size'], device=error_t_lig.device)

        # The KL between q(zT | x) and p(zT) = Normal(0, 1).
        # Should be close to zero.
        kl_prior = self.kl_prior(xh0_lig, ligand['mask'], ligand['size'])

        if self.training:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(
                    ligand, z_t_lig, eps_t_lig, net_out_lig, gamma_t)

            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand * \
                              t_is_zero.squeeze()
            loss_0_h = -log_ph_given_z0 * t_is_zero.squeeze()

            # apply t_is_zero mask
            error_t_lig = error_t_lig * t_is_not_zero.squeeze()

        else:
            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), ligand['x'])

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            z_0_lig, xh_pocket, eps_0_lig = \
                self.noised_representation(xh0_lig, xh0_pocket, ligand['mask'],
                                           pocket['mask'], gamma_0)

            net_out_0_lig, _ = self.dynamics(
                z_0_lig, xh_pocket, t_zeros, ligand['mask'], pocket['mask'])

            log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(
                    ligand, z_0_lig, eps_0_lig, net_out_0_lig, gamma_0)
            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand
            loss_0_h = -log_ph_given_z0

        # sample size prior
        log_pN = self.log_pN(ligand['size'], pocket['size'])

        info = {
            'eps_hat_lig_x': scatter_mean(
                net_out_lig[:, :self.n_dims].abs().mean(1), ligand['mask'],
                dim=0).mean(),
            'eps_hat_lig_h': scatter_mean(
                net_out_lig[:, self.n_dims:].abs().mean(1), ligand['mask'],
                dim=0).mean(),
        }
        loss_terms = (delta_log_px, error_t_lig, torch.tensor(0.0), SNR_weight,
                      loss_0_x_ligand, torch.tensor(0.0), loss_0_h,
                      neg_log_constants, kl_prior, log_pN,
                      t_int.squeeze(), xh_lig_hat)
        return (*loss_terms, info) if return_info else loss_terms
    
    def partially_noised_ligand(self, ligand, pocket, noising_steps):
        """
        Partially noises a ligand to be later denoised.
        """

        # Inflate timestep into an array
        t_int = torch.ones(size=(ligand['size'].size(0), 1),
            device=ligand['x'].device).float() * noising_steps

        # Normalize t to [0, 1].
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand['x'])

        # Concatenate x, and h[categorical].
        xh0_lig = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        # Center the input nodes
        xh0_lig[:, :self.n_dims], xh0_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(xh0_lig[:, :self.n_dims],
                                   xh0_pocket[:, :self.n_dims],
                                   ligand['mask'], pocket['mask'])

        # Find noised representation
        z_t_lig, xh_pocket, eps_t_lig = \
            self.noised_representation(xh0_lig, xh0_pocket, ligand['mask'],
                                       pocket['mask'], gamma_t)
            
        return z_t_lig, xh_pocket, eps_t_lig

    def diversify(self, ligand, pocket, noising_steps):
        """
        Diversifies a set of ligands via noise-denoising
        """

        # Normalize data, take into account volume change in x.
        ligand, pocket = self.normalize(ligand, pocket)

        z_lig, xh_pocket, _ = self.partially_noised_ligand(ligand, pocket, noising_steps)

        timesteps = self.T
        n_samples = len(pocket['size'])
        device = pocket['x'].device

        # xh0_pocket is the original pocket while xh_pocket might be a
        # translated version of it
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        lig_mask = ligand['mask']

        self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.

        for s in reversed(range(0, noising_steps)):
            s_array = torch.full((n_samples, 1), fill_value=s,
                                 device=z_lig.device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps

            z_lig, xh_pocket = self.sample_p_zs_given_zt(
                s_array, t_array, z_lig.detach(), xh_pocket.detach(), lig_mask, pocket['mask'])

        # Finally sample p(x, h | z_0).
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket, lig_mask, pocket['mask'], n_samples)

        self.assert_mean_zero_with_mask(x_lig, lig_mask)

        # Overwrite last frame with the resulting x and h.
        out_lig = torch.cat([x_lig, h_lig], dim=1)
        out_pocket = torch.cat([x_pocket, h_pocket], dim=1)

        # remove frame dimension if only the final molecule is returned
        return out_lig, out_pocket, lig_mask, pocket['mask']


    def xh_given_zt_and_epsilon(self, z_t, epsilon, gamma_t, batch_mask):
        """ Equation (7) in the EDM paper """
        alpha_t = self.alpha(gamma_t, z_t)
        sigma_t = self.sigma(gamma_t, z_t)
        xh = z_t / alpha_t[batch_mask] - epsilon * sigma_t[batch_mask] / \
             alpha_t[batch_mask]
        return xh

    def sample_p_zt_given_zs(self, zs_lig, xh0_pocket, ligand_mask, pocket_mask,
                             gamma_t, gamma_s, fix_noise=False):
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zs_lig)

        mu_lig = alpha_t_given_s[ligand_mask] * zs_lig
        zt_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma_t_given_s, ligand_mask, pocket_mask,
            fix_noise)

        return zt_lig, xh0_pocket

    def sample_p_zs_given_zt(self, s, t, zt_lig, xh0_pocket, ligand_mask,
                             pocket_mask, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)

        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)

        # Neural net prediction.
        eps_t_lig, _ = self.dynamics(
            zt_lig, xh0_pocket, t, ligand_mask, pocket_mask)

        # Compute mu for p(zs | zt).
        # Note: mu_{t->s} = 1 / alpha_{t|s} z_t - sigma_{t|s}^2 / sigma_t / alpha_{t|s} epsilon
        # follows from the definition of mu_{t->s} and Equ. (7) in the EDM paper
        mu_lig = zt_lig / alpha_t_given_s[ligand_mask] - \
                 (sigma2_t_given_s / alpha_t_given_s / sigma_t)[ligand_mask] * \
                 eps_t_lig

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the parameters derived from zt.
        zs_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, ligand_mask, pocket_mask, fix_noise)

        self.assert_mean_zero_with_mask(zt_lig[:, :self.n_dims], ligand_mask)

        return zs_lig, xh0_pocket

    def sample_p_zs_given_zt_rl(self, s, t, zt_lig, xh0_pocket, ligand_mask,
                                pocket_mask, action=None, fix_noise=False,
                                mu_via_x0=True):
        """
        RL-aware version: sample (or evaluate) zs ~ p(z_s | z_t) and return
        distribution stats + log prob of provided action.

        Args:
            s, t: tensors (batch, 1) with normalized timesteps s=t-1.
            zt_lig: (N_lig_total, n_dims + atom_nf)
            xh0_pocket: (N_pocket_total, n_dims + residue_nf)
            ligand_mask: (N_lig_total,) batch index per ligand atom
            pocket_mask: (N_pocket_total,) batch index per pocket residue
            action: Optional pre-chosen zs_lig sample (same shape as zt_lig).
            fix_noise: unused (kept for interface parity)
            mu_via_x0: if True use x0 prediction parameterization for mean.

        Returns:
            dict with:
              action: sampled (or provided) zs_lig (before COM projection)
              mu: mean for each ligand atom
              sigma: per-batch scalar std (broadcasted by mask)
              eps_t: network epsilon prediction
              log_prob: per-sample log prob (normalized per atom)
              zs_xzeromean: action with x part projected to COM-free together with pocket
              action_entropy: per-sample entropy (normalized per atom)
        """
        # gamma and conditional coefficients
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)

        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)       # (batch,1)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)       # (batch,1)

        # Network epsilon prediction
        eps_t_lig, _ = self.dynamics(zt_lig, xh0_pocket, t, ligand_mask, pocket_mask)

        # Optionally compute mu via predicted x0
        if mu_via_x0:
            alpha_t = self.alpha(gamma_t, target_tensor=zt_lig)   # (batch,1)
            alpha_s = self.alpha(gamma_s, target_tensor=zt_lig)   # (batch,1)
            pred_x0 = (zt_lig - sigma_t[ligand_mask] * eps_t_lig) / alpha_t[ligand_mask]
            # Clamp only h-part (categorical logits) to [0,1] to avoid runaway
            pred_x0 = torch.cat(
                [pred_x0[:, :self.n_dims],
                 pred_x0[:, self.n_dims:].clamp(0.0, 1.0)],
                dim=1
            )
            # mu formula (broadcast batch scalars to atoms via ligand_mask)
            sigma_s_l = sigma_s[ligand_mask]
            sigma_t_l = sigma_t[ligand_mask]
            alpha_t_given_s_l = alpha_t_given_s[ligand_mask]
            sigma2_t_given_s_l = sigma2_t_given_s[ligand_mask]
            alpha_s_l = alpha_s[ligand_mask]

            mu_lig = (alpha_t_given_s_l * (sigma_s_l ** 2) / (sigma_t_l ** 2) * zt_lig +
                      alpha_s_l * sigma2_t_given_s_l / (sigma_t_l ** 2) * pred_x0)
        else:
            alpha_t_given_s_l = alpha_t_given_s[ligand_mask]
            sigma2_t_given_s_l = sigma2_t_given_s[ligand_mask]
            sigma_t_l = sigma_t[ligand_mask]
            mu_lig = (zt_lig / alpha_t_given_s_l -
                      (sigma2_t_given_s_l / alpha_t_given_s_l / sigma_t_l) * eps_t_lig)

        # Sigma for p(z_s | z_t)
        sigma = sigma_t_given_s * sigma_s / sigma_t              # (batch,1)
        sigma_lig = sigma[ligand_mask]                           # (N_atoms,1)

        # Sample action if not provided
        if action is None:
            noise = self.sample_gaussian(size=mu_lig.shape, device=mu_lig.device)
            action = mu_lig + sigma_lig * noise

        # Keep a zero-COM x projection version (ligand + pocket recentered)
        # Clone to avoid modifying original action when projecting
        zs_lig = action.clone()
        zs_lig[:, :self.n_dims], xh0_pocket[:, :self.n_dims]  = self.remove_mean_batch(
            zs_lig[:, :self.n_dims],
            xh0_pocket[:, :self.n_dims],
            ligand_mask, pocket_mask
        )

        # Log prob of action under Normal(mu, sigma)
        # Broadcast sigma_lig over feature dimension
        log_prob_feat = - (action - mu_lig) ** 2 / (2 * (sigma_lig ** 2)) \
                        - torch.log(sigma_lig) \
                        - 0.5 * math.log(2 * math.pi)
        # Sum over feature dims -> per-node
        log_prob_node = log_prob_feat.sum(dim=1)
        # Aggregate per batch
        log_prob_batch = scatter_add(log_prob_node, ligand_mask, dim=0)
        n_atoms = scatter_add(torch.ones_like(log_prob_node), ligand_mask, dim=0)
        log_prob = log_prob_batch / n_atoms

        # Entropy (same scalar per feature, sum over features, normalize)
        entropy_feat = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(sigma_lig)
        entropy_node = entropy_feat.sum(dim=1)
        entropy_batch = scatter_add(entropy_node, ligand_mask, dim=0) / n_atoms

        return {
            "action": action,
            "mu": mu_lig,
            "sigma": sigma_lig,
            "eps_t": eps_t_lig,
            "log_prob": log_prob,
            "zs_lig": zs_lig,
            "xh0_pocket": xh0_pocket,
            "action_entrophy": entropy_batch
        }

    def sample_p_stepscale_given_zt_rl(self, t, zt_lig, xh_pocket,
                                  ligand_mask, pocket_mask, action=None,
                                  fix_noise=False, mu_via_x0=True):
        """
        RL-aware version: sample (or evaluate) zs ~ p(z_s | z_t) with s=t-1.

        See sample_p_zs_given_zt_rl() for details.
        """
        from torch.distributions import Beta as TorchBeta
        ab = self.s_predictor(zt_lig, xh_pocket, t, ligand_mask, pocket_mask)
        alpha = ab[:, 0].clamp_min(1e-6)
        beta = ab[:, 1].clamp_min(1e-6)
        dist = TorchBeta(alpha, beta)
        if action is None:
            action = dist.sample().unsqueeze(1) # action is in range (0,1)
            # action = (dist.rsample() if use_reparam else dist.sample()).unsqueeze(1)
        action = action.clamp(1e-3, 1 - 1e-3)
        log_prob = dist.log_prob(action.squeeze(1))

        # s = t - time_interval_scale * self.inference_interval / self.T
        out_dict = {
            # "s": s,
            "action2": action,
            "log_prob2": log_prob,
            "ab": ab,
        }
        return out_dict


    def sample_combined_position_feature_noise(self, lig_indices, xh0_pocket,
                                               pocket_indices):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise
        for z_h.
        """
        raise NotImplementedError("Use sample_normal_zero_com() instead.")

    def sample(self, *args):
        raise NotImplementedError("Conditional model does not support sampling "
                                  "without given pocket.")

    @torch.no_grad()
    def sample_given_pocket(self, pocket, num_nodes_lig, return_frames=1,
                            timesteps=None):
        """
        Draw samples from the generative model. Optionally, return intermediate
        states for visualization purposes.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0

        n_samples = len(pocket['size'])
        device = pocket['x'].device

        _, pocket = self.normalize(pocket=pocket)

        # xh0_pocket is the original pocket while xh_pocket might be a
        # translated version of it
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        lig_mask = utils.num_nodes_to_batch_mask(
            n_samples, num_nodes_lig, device)

        # Sample from Normal distribution in the pocket center
        mu_lig_x = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
        sigma = torch.ones_like(pocket['size']).unsqueeze(1)

        z_lig, xh_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, lig_mask, pocket['mask'])

        self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

        out_lig = torch.zeros((return_frames,) + z_lig.size(),
                              device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket.size(),
                                 device=device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, timesteps)):
            s_array = torch.full((n_samples, 1), fill_value=s,
                                 device=z_lig.device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps

            z_lig, xh_pocket = self.sample_p_zs_given_zt(
                s_array, t_array, z_lig, xh_pocket, lig_mask, pocket['mask'])

            # save frame
            if (s * return_frames) % timesteps == 0:
                idx = (s * return_frames) // timesteps
                out_lig[idx], out_pocket[idx] = \
                    self.unnormalize_z(z_lig, xh_pocket)

        # Finally sample p(x, h | z_0).
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket, lig_mask, pocket['mask'], n_samples)

        self.assert_mean_zero_with_mask(x_lig, lig_mask)

        # Correct CoM drift for examples without intermediate states
        if return_frames == 1:
            max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f'Warning CoG drift with error {max_cog:.3f}. Projecting '
                      f'the positions down.')
                x_lig, x_pocket = self.remove_mean_batch(
                    x_lig, x_pocket, lig_mask, pocket['mask'])

        # Overwrite last frame with the resulting x and h.
        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)

        # remove frame dimension if only the final molecule is returned
        return out_lig.squeeze(0), out_pocket.squeeze(0), lig_mask, \
               pocket['mask']

    @torch.no_grad()
    def inpaint(self, ligand, pocket, lig_fixed, resamplings=1, return_frames=1,
                timesteps=None, center='ligand'):
        """
        Draw samples from the generative model while fixing parts of the input.
        Optionally, return intermediate states for visualization purposes.
        Inspired by Algorithm 1 in:
        Lugmayr, Andreas, et al.
        "Repaint: Inpainting using denoising diffusion probabilistic models."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
        Recognition. 2022.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0

        if len(lig_fixed.size()) == 1:
            lig_fixed = lig_fixed.unsqueeze(1)

        n_samples = len(ligand['size'])
        device = pocket['x'].device

        # Normalize
        ligand, pocket = self.normalize(ligand, pocket)

        # xh0_pocket is the original pocket while xh_pocket might be a
        # translated version of it
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        com_pocket_0 = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        xh0_ligand = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
        xh_ligand = xh0_ligand.clone()

        # Center initial system, subtract COM of known parts
        if center == 'ligand':
            mean_known = scatter_mean(ligand['x'][lig_fixed.bool().view(-1)],
                                      ligand['mask'][lig_fixed.bool().view(-1)],
                                      dim=0)
        elif center == 'pocket':
            mean_known = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        else:
            raise NotImplementedError(
                f"Centering option {center} not implemented")

        # Sample from Normal distribution in the ligand center
        mu_lig_x = mean_known
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[ligand['mask']]
        sigma = torch.ones_like(pocket['size']).unsqueeze(1)

        z_lig, xh_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, ligand['mask'], pocket['mask'])

        # Output tensors
        out_lig = torch.zeros((return_frames,) + z_lig.size(),
                              device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket.size(),
                                 device=device)

        # Iteratively sample with resampling iterations
        for s in reversed(range(0, timesteps)):

            # resampling iterations
            for u in range(resamplings):

                # Denoise one time step: t -> s
                s_array = torch.full((n_samples, 1), fill_value=s,
                                     device=device)
                t_array = s_array + 1
                s_array = s_array / timesteps
                t_array = t_array / timesteps

                gamma_t = self.gamma(t_array)
                gamma_s = self.gamma(s_array)

                # sample inpainted part
                z_lig_unknown, xh_pocket = self.sample_p_zs_given_zt(
                    s_array, t_array, z_lig, xh_pocket, ligand['mask'],
                    pocket['mask'])

                # sample known nodes from the input
                com_pocket = scatter_mean(xh_pocket[:, :self.n_dims],
                                          pocket['mask'], dim=0)
                xh_ligand[:, :self.n_dims] = \
                    ligand['x'] + (com_pocket - com_pocket_0)[ligand['mask']]
                z_lig_known, xh_pocket, _ = self.noised_representation(
                    xh_ligand, xh_pocket, ligand['mask'], pocket['mask'],
                    gamma_s)

                # move center of mass of the noised part to the center of mass
                # of the corresponding denoised part before combining them
                # -> the resulting system should be COM-free
                com_noised = scatter_mean(
                    z_lig_known[lig_fixed.bool().view(-1)][:, :self.n_dims],
                    ligand['mask'][lig_fixed.bool().view(-1)], dim=0)
                com_denoised = scatter_mean(
                    z_lig_unknown[lig_fixed.bool().view(-1)][:, :self.n_dims],
                    ligand['mask'][lig_fixed.bool().view(-1)], dim=0)
                dx = com_denoised - com_noised
                z_lig_known[:, :self.n_dims] = z_lig_known[:, :self.n_dims] + dx[ligand['mask']]
                xh_pocket[:, :self.n_dims] = xh_pocket[:, :self.n_dims] + dx[pocket['mask']]

                # combine
                z_lig = z_lig_known * lig_fixed + z_lig_unknown * (
                            1 - lig_fixed)

                if u < resamplings - 1:
                    # Noise the sample
                    z_lig, xh_pocket = self.sample_p_zt_given_zs(
                        z_lig, xh_pocket, ligand['mask'], pocket['mask'],
                        gamma_t, gamma_s)

                # save frame at the end of a resampling cycle
                if u == resamplings - 1:
                    if (s * return_frames) % timesteps == 0:
                        idx = (s * return_frames) // timesteps

                        out_lig[idx], out_pocket[idx] = \
                            self.unnormalize_z(z_lig, xh_pocket)

        # Finally sample p(x, h | z_0).
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket, ligand['mask'], pocket['mask'], n_samples)

        # Overwrite last frame with the resulting x and h.
        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)

        # remove frame dimension if only the final molecule is returned
        return out_lig.squeeze(0), out_pocket.squeeze(0), ligand['mask'], \
               pocket['mask']

    @classmethod
    def remove_mean_batch(cls, x_lig, x_pocket, lig_indices, pocket_indices):

        # Just subtract the center of mass of the sampled part
        mean = scatter_mean(x_lig, lig_indices, dim=0)

        x_lig = x_lig - mean[lig_indices]
        x_pocket = x_pocket - mean[pocket_indices]
        return x_lig, x_pocket

    def rl_episode_given_pocket(self, pocket, num_nodes_lig, pocket_com_before=None, return_frames=1, timesteps=None, inference_interval=1, ppo_config=None, optimizer=None):
        """ 
        RL rollout and (optional) PPO update for conditional diffusion.
        - Collects a trajectory with obs, actions, log_probs at a stride defined by
          ppo_config.inference_interval.
        - Uses sample_p_zs_given_zt_rl to produce actions and log-probs.
        - At the end, samples x,h | z0 and (optionally) computes rewards via
          ppo_config.reward_fn(x_lig, h_lig, lig_mask, pocket) -> (n_samples,)
        - If `optimizer` is provided, performs PPO updates using the collected
          trajectory and advantages.

        Returns:
            episodic_metrics: dict with at least 'rewards' (mean), 'advantages' (mean),
                              'valid_rate' if reward_fn returns such stats, else omitted.
        """
        assert ppo_config is not None, "ppo_config is required"
        assert hasattr(ppo_config, "inference_interval") and ppo_config.inference_interval > 0
        timesteps = self.T if timesteps is None else timesteps
        assert timesteps % ppo_config.inference_interval == 0, \
            "timesteps must be divisible by inference_interval"
        assert 0 < return_frames <= timesteps and timesteps % return_frames == 0

        n_samples = len(pocket['size'])
        device = pocket['x'].device

        # Normalize pocket only (as in sample_given_pocket)
        _, pocket = self.normalize(pocket=pocket)

        # Prepare pocket features and masks
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        lig_mask = utils.num_nodes_to_batch_mask(n_samples, num_nodes_lig, device)

        # Initialize ligand latent z in pocket center and COM-free
        mu_lig_x = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
        sigma = torch.ones_like(pocket['size']).unsqueeze(1)

        z_lig, xh_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, lig_mask, pocket['mask']
        )
        self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

        # For optional visualization frames (unnormalized z for intermediate steps)
        out_lig = torch.zeros((return_frames,) + z_lig.size(), device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket.size(), device=device)

        # Episode buffers (time-major)
        episode_length = timesteps // ppo_config.inference_interval
        obs_steps = torch.zeros((episode_length,) + z_lig.size(), device=z_lig.device)
        action_steps = torch.zeros((episode_length,) + z_lig.size(), device=z_lig.device)
        log_prob_steps = torch.zeros((episode_length, n_samples), device=z_lig.device)
        t_steps = torch.zeros((episode_length, n_samples, 1), device=z_lig.device)
        s_steps = torch.zeros((episode_length, n_samples, 1), device=z_lig.device)
        done_steps = torch.zeros((episode_length, n_samples), device=z_lig.device)
        # Store xh_pocket per step (needed to recompute log_probs during PPO update)
        xh_pocket_steps = torch.zeros((episode_length,) + xh_pocket.size(), device=device)

        next_done = torch.zeros(n_samples, device=z_lig.device)
        if ppo_config.predict_s:
            action_time_interval_scale_steps = torch.zeros((episode_length, n_samples, 1), device=z_lig.device)
            log_prob_time_interval_scale_steps = torch.zeros((episode_length, n_samples), device=z_lig.device)

        # Reverse diffusion with stride = inference_interval
        frame_stride = timesteps // return_frames
        flex_t = torch.full((n_samples, 1), fill_value=timesteps, device=z_lig.device)
        flex_t_norm = flex_t / timesteps
        for idx, s in enumerate(reversed(range(0, timesteps, ppo_config.inference_interval))):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z_lig.device)
            t_array = s_array + ppo_config.inference_interval

            regular_s_norm = s_array / timesteps
            regular_t_norm = t_array / timesteps

            # Record observation and dones
            obs_steps[idx] = z_lig
            xh_pocket_steps[idx] = xh_pocket
            done_steps[idx] = next_done

            with torch.no_grad():
                if ppo_config.predict_s:
                    out_dict2 = self.sample_p_stepscale_given_zt_rl(flex_t_norm, zt_lig=z_lig, xh_pocket=xh_pocket,
                    ligand_mask=lig_mask, pocket_mask=pocket['mask'],
                    mu_via_x0=True)
                    flex_s_norm = regular_t_norm - out_dict2['action2'] * ppo_config.inference_interval / timesteps

                    out_dict = self.sample_p_zs_given_zt_rl(
                    flex_s_norm, flex_t_norm, zt_lig=z_lig, xh0_pocket=xh_pocket,
                    ligand_mask=lig_mask, pocket_mask=pocket['mask'],
                    mu_via_x0=True
                    )

                    s_steps[idx] = flex_s_norm
                    t_steps[idx] = flex_t_norm
                    flex_t_norm = flex_s_norm
                    action_time_interval_scale_steps[idx] = out_dict2['action2']
                    log_prob_time_interval_scale_steps[idx] = out_dict2['log_prob2']
                    
                else: 
                    out_dict = self.sample_p_zs_given_zt_rl(
                    regular_s_norm, regular_t_norm, zt_lig=z_lig, xh0_pocket=xh_pocket,
                    ligand_mask=lig_mask, pocket_mask=pocket['mask'],
                    mu_via_x0=True
                    )

                    s_steps[idx] = regular_s_norm
                    t_steps[idx] = regular_t_norm

            # Step environment (zs_xzeromean is zero-COM version of the action)
            z_lig = out_dict['zs_lig']
            xh_pocket = out_dict['xh0_pocket']
            action_steps[idx] = out_dict['action']
            log_prob_steps[idx] = out_dict['log_prob']

            # Mark done only at the final step
            if s == 0:
                next_done = torch.ones(n_samples, device=z_lig.device)

            # Save frame if requested (unnormalized z for viz)
            if (s % frame_stride) == 0:
                frame_idx = s // frame_stride
                out_lig[frame_idx], out_pocket[frame_idx] = self.unnormalize_z(z_lig, xh_pocket)

            # Keep COM-free in x-part
            self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

        # Final sample p(x, h | z_0)
        if ppo_config.predict_s:
            x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_zt(
            z_lig, xh_pocket, lig_mask, pocket['mask'], flex_s_norm,
        )
        else:
            x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
                z_lig, xh_pocket, lig_mask, pocket['mask'], n_samples
            )
        # x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
        #     z_lig, xh_pocket, lig_mask, pocket['mask'], n_samples
        # )
        self.assert_mean_zero_with_mask(x_lig, lig_mask)

        # Correct CoM drift if no intermediate frames
        if return_frames == 1:
            max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f'Warning CoG drift with error {max_cog:.3f}. Projecting the positions down.')
                x_lig, x_pocket = self.remove_mean_batch(x_lig, x_pocket, lig_mask, pocket['mask'])

        # Overwrite last frame
        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)

        rollout_buffer = SimpleNamespace(obs_steps=obs_steps, action_steps=action_steps,
                                        log_prob_steps=log_prob_steps,
                                        s_steps=s_steps,
                                        t_steps=t_steps,
                                        xh_pocket_steps=xh_pocket_steps,
                                        done_steps=done_steps,
                                        log_prob_time_interval_scale_steps=log_prob_time_interval_scale_steps if ppo_config.predict_s else None,
                                        action_time_interval_scale_steps=action_time_interval_scale_steps if ppo_config.predict_s else None
                                        )
        
        return out_lig.squeeze(0), out_pocket.squeeze(0), lig_mask, \
               pocket['mask'], rollout_buffer



# ------------------------------------------------------------------------------
# The same model without subspace-trick
# ------------------------------------------------------------------------------
class SimpleConditionalDDPM(ConditionalDDPM):
    """
    Simpler conditional diffusion module without subspace-trick.
    - rotational equivariance is guaranteed by construction
    - translationally equivariant likelihood is achieved by first mapping
      samples to a space where the context is COM-free and evaluating the
      likelihood there
    - molecule generation is equivariant because we can first sample in the
      space where the context is COM-free and translate the whole system back to
      the original position of the context later
    """
    def subspace_dimensionality(self, input_size):
        """ Override because we don't use the linear subspace anymore. """
        return input_size * self.n_dims

    @classmethod
    def remove_mean_batch(cls, x_lig, x_pocket, lig_indices, pocket_indices):
        """ Hacky way of removing the centering steps without changing too much
        code. """
        return x_lig, x_pocket

    @staticmethod
    def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
        return

    def forward(self, ligand, pocket, return_info=False):

        # Subtract pocket center of mass
        pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        ligand['x'] = ligand['x'] - pocket_com[ligand['mask']]
        pocket['x'] = pocket['x'] - pocket_com[pocket['mask']]

        return super(SimpleConditionalDDPM, self).forward(
            ligand, pocket, return_info)

    @torch.no_grad()
    def sample_given_pocket(self, pocket, num_nodes_lig, return_frames=1,
                            timesteps=None):

        # Subtract pocket center of mass
        pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        pocket['x'] = pocket['x'] - pocket_com[pocket['mask']]

        return super(SimpleConditionalDDPM, self).sample_given_pocket(
            pocket, num_nodes_lig, return_frames, timesteps)
