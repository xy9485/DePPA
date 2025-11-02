import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np
from equivariant_diffusion.egnn_new import EGNN

class SPredictor(nn.Module):
    """
    Rotation-invariant s-predictor producing Beta(alpha, beta) params for r in (0,1).

    forward(zt_lig, xh0_pocket, t, ligand_mask, pocket_mask) -> (B, 2)

    Invariance strategy:
      - Uses only rotation-invariant geometry: pairwise distances and radii of gyration.
      - Pools scalar node features (categorical logits/one-hot) via mean.
    """

    def __init__(self, atom_nf: int, residue_nf: int, n_dims: int, hidden_dim: int = 128):
        super().__init__()
        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.n_dims = n_dims

        # Feature vector per graph/batch:
        #   1 (t)
        # + atom_nf mean pooled ligand features
        # + residue_nf mean pooled pocket features
        # + 10 geometric scalars (means/stds/medians + radii of gyration)
        input_dim = 1 + atom_nf + residue_nf + 10

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

        self.eps = 1e-6

    @staticmethod
    def _pairwise_stats(x_a: torch.Tensor, x_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x_a.device
        if x_a.numel() == 0 or x_b.numel() == 0:
            z = torch.tensor(0.0, device=device)
            return z, z, z
        d = torch.cdist(x_a, x_b)
        if d.numel() == 0:
            z = torch.tensor(0.0, device=device)
            return z, z, z
        flat = d.reshape(-1)
        return flat.mean(), flat.std(unbiased=False), flat.median()

    @staticmethod
    def _within_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        n = x.shape[0]
        if n < 2:
            z = torch.tensor(0.0, device=device)
            return z, z, z
        d = torch.cdist(x, x)
        iu = torch.triu_indices(n, n, offset=1)
        vals = d[iu[0], iu[1]]
        if vals.numel() == 0:
            z = torch.tensor(0.0, device=device)
            return z, z, z
        return vals.mean(), vals.std(unbiased=False), vals.median()

    @staticmethod
    def _radius_of_gyration(x: torch.Tensor) -> torch.Tensor:
        device = x.device
        if x.numel() == 0:
            return torch.tensor(0.0, device=device)
        com = x.mean(dim=0, keepdim=True)
        r2 = ((x - com) ** 2).sum(dim=1)
        return (r2.mean() + 1e-12).sqrt()

    def forward(self, zt_lig: torch.Tensor, xh0_pocket: torch.Tensor,
                t: torch.Tensor, ligand_mask: torch.Tensor,
                pocket_mask: torch.Tensor) -> torch.Tensor:
        x_l = zt_lig[:, :self.n_dims]
        h_l = zt_lig[:, self.n_dims:]
        x_p = xh0_pocket[:, :self.n_dims]
        h_p = xh0_pocket[:, self.n_dims:]

        batch_size = t.shape[0]
        feats = []
        for b in range(batch_size):
            lm = (ligand_mask == b)
            pm = (pocket_mask == b)

            xlb = x_l[lm]
            xpb = x_p[pm]
            hlb = h_l[lm]
            hpb = h_p[pm]

            h_l_mean = hlb.mean(dim=0) if hlb.numel() > 0 else torch.zeros(self.atom_nf, device=x_l.device, dtype=x_l.dtype)
            h_p_mean = hpb.mean(dim=0) if hpb.numel() > 0 else torch.zeros(self.residue_nf, device=x_l.device, dtype=x_l.dtype)

            ll_mean, ll_std, ll_med = self._within_stats(xlb)
            pp_mean, pp_std, pp_med = self._within_stats(xpb)
            lp_mean, lp_std, lp_med = self._pairwise_stats(xlb, xpb)
            rg_l = self._radius_of_gyration(xlb)
            rg_p = self._radius_of_gyration(xpb)

            f = torch.cat([
                t[b].reshape(-1),
                h_l_mean.reshape(-1),
                h_p_mean.reshape(-1),
                torch.stack([ll_mean, ll_std, pp_mean, pp_std,
                             lp_mean, lp_std, rg_l, rg_p, ll_med, lp_med])
            ])
            feats.append(f)

        feat_batch = torch.stack(feats, dim=0)
        out = self.mlp(feat_batch)
        return F.softplus(out) + self.eps  # (B,2) positive alpha,beta


@torch.no_grad()
def predict_s_params(s_predictor: SPredictor, zt_lig: torch.Tensor, xh0_pocket: torch.Tensor,
                     t: torch.Tensor, ligand_mask: torch.Tensor, pocket_mask: torch.Tensor) -> torch.Tensor:
    """Free function to get Beta(alpha, beta) without touching the DDPM class."""
    return s_predictor(zt_lig, xh0_pocket, t, ligand_mask, pocket_mask)


@torch.no_grad()
def sample_time_interval_scale(s_predictor: SPredictor, zt_lig: torch.Tensor, xh_pocket: torch.Tensor,
             t: torch.Tensor, ligand_mask: torch.Tensor, pocket_mask: torch.Tensor,
             action: None, use_reparam: bool = False):
    from torch.distributions import Beta as TorchBeta
    ab = s_predictor(zt_lig, xh_pocket, t, ligand_mask, pocket_mask)
    alpha = ab[:, 0].clamp_min(1e-6)
    beta = ab[:, 1].clamp_min(1e-6)
    dist = TorchBeta(alpha, beta)
    if action is None:
        action = (dist.rsample() if use_reparam else dist.sample()).unsqueeze(1)
    time_interval_scale = action.clamp(1e-6, 1 - 1e-6)
    # s = r * t

    # compute log_prob of r
    log_prob = dist.log_prob(action.squeeze(1))

    out_dict = {
        # "s": s,
        "action": time_interval_scale,
        "log_prob": log_prob,
        "ab": ab
    }

    return out_dict


class EGNNSPredictor(nn.Module):
    """
    EGNN-based invariant s-predictor that mirrors EGNNDynamics' architecture
    (encoders, EGNN trunk, edge handling, time conditioning) but WITHOUT
    computing velocities (no x_final - x). Instead, it adds an invariant
    readout over the x part and pooled scalar features to produce Beta
    parameters (alpha, beta) for r in (0,1).

    Forward signature matches EGNNDynamics except it returns (batch,2):
        forward(xh_atoms, xh_residues, t, mask_atoms, mask_residues) -> alpha_beta
    """

    def __init__(self, atom_encoder, residue_encoder, backbone, n_dims=3, h_nf=128,
                 device='cpu', condition_time=True,
                 update_pocket_coords=True, edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None, edge_cutoff_interaction=None,
                 edge_embedding_dim=None,
                 readout_hidden_dim=128):
        super().__init__()
        self.n_dims = n_dims
        self.device = device
        self.condition_time = condition_time
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_nf = edge_embedding_dim
        self.update_pocket_coords = update_pocket_coords
        self.h_nf = h_nf

        self.atom_encoder = atom_encoder
        self.residue_encoder = residue_encoder
        self.egnn = backbone


        self.edge_embedding = nn.Embedding(3, self.edge_nf) \
            if self.edge_nf is not None else None
        self.edge_nf = 0 if self.edge_nf is None else self.edge_nf

        self._eps = 1.0

        # Pairwise invariant readout: sum_{ij in edges} f(h_i, h_j, rbf(|x_i-x_j|)) -> R^2
        rbf_K = 32
        # self._pair_hdim = joint_nf  # scalar channels after EGNN (time removed before readout)
        self.rbf = RBF(K=rbf_K, r_cut=8.0)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.h_nf + rbf_K, readout_hidden_dim),
            nn.SiLU(),
            nn.Linear(readout_hidden_dim, readout_hidden_dim),
            nn.SiLU(),
            nn.Linear(readout_hidden_dim, 2)
        )

    def get_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket):
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)

        if self.edge_cutoff_p is not None:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)

        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)

        adj = torch.cat((torch.cat((adj_ligand, adj_cross), dim=1),
                         torch.cat((adj_cross.T, adj_pocket), dim=1)), dim=0)
        edges = torch.stack(torch.where(adj), dim=0)

        return edges

    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues):
        # Split inputs
        x_atoms = xh_atoms[:, :self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims:].clone()
        x_residues = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()

        # Encode to joint space
        h_atoms = self.atom_encoder(h_atoms)
        h_residues = self.residue_encoder(h_residues)

        # Combine and time-condition
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                h_time = t[mask]
            h = torch.cat([h, h_time], dim=1)

        # Edges and edge types
        edges = self.get_edges(mask_atoms, mask_residues, x_atoms, x_residues)
        assert torch.all(mask[edges[0]] == mask[edges[1]])

        if self.edge_nf > 0:
            edge_types = torch.zeros(edges.size(1), dtype=int, device=edges.device)
            edge_types[(edges[0] < len(mask_atoms)) & (edges[1] < len(mask_atoms))] = 1
            edge_types[(edges[0] >= len(mask_atoms)) & (edges[1] >= len(mask_atoms))] = 2
            edge_types = self.edge_embedding(edge_types)
        else:
            edge_types = None

        # EGNN trunk (no velocity computation)
        update_coords_mask = None if self.update_pocket_coords \
            else torch.cat((torch.ones_like(mask_atoms),
                            torch.zeros_like(mask_residues))).unsqueeze(1)
        h_final, x_final = self.egnn(h, x, edges,
                                     update_coords_mask=update_coords_mask,
                                     batch_mask=mask, edge_attr=edge_types)

        #raise exception if h_final or x_final contains nan
        if torch.isnan(h_final).any() or torch.isnan(x_final).any():
            raise ValueError("NaN detected in h_final or x_final in EGNNSPredictor forward pass.")

        # Invariant edge readout over final embeddings and coordinates
        # Use joint-space scalars (h_final) for pair features
        src, dst = edges  # [E]
        if src.numel() == 0:
            # No edges -> return small positive defaults per batch
            batch_size = t.shape[0]
            return F.softplus(h.new_zeros((batch_size, 2))) + self._eps

        rij = x_final[src] - x_final[dst]                # [E, n_dims]
        r = torch.linalg.norm(rij, dim=-1)               # [E]
        phi_r = self.rbf(r)                              # [E, K]
        h_i = h_final[src]
        h_j = h_final[dst]
        edge_feat = torch.cat([h_i, h_j, phi_r], dim=-1) # [E, 2*H + K]
        edge_out = self.edge_mlp(edge_feat)              # [E, 2]

        # Sum per batch over edges (batch index given by node batch of src)
        batch_idx_e = mask[src]                          # [E]
        batch_size = t.shape[0]
        out = h.new_zeros((batch_size, 2))
        out.index_add_(0, batch_idx_e, edge_out)

        # Map to positive Beta parameters
        return F.softplus(out) + self._eps



# class RBF(nn.Module):
#     def __init__(self, centers, gamma, cutoff):
#         super().__init__()
#         self.register_buffer('c', torch.tensor(centers).float())
#         self.gamma = gamma
#         self.cutoff = cutoff
#     def forward(self, r):
#         # smooth C2 envelope
#         x = torch.clamp(1 - (r / self.cutoff), min=0.0)
#         env = x**2 * (3 - 2*x)             # [E]
#         rbf = torch.exp(-self.gamma * (r.unsqueeze(-1) - self.c)**2)  # [E, K]
#         return env.unsqueeze(-1) * rbf      # [E, K]


class RBF(nn.Module):
    def __init__(self, K=64, r_cut=8.0, gamma=None):
        super().__init__()
        centers = torch.linspace(0.0, r_cut, K)
        self.register_buffer("centers", centers)
        self.r_cut = r_cut
        dr = centers[1] - centers[0]
        self.gamma = (1.0 / (dr + 1e-8)**2) if gamma is None else gamma

    def forward(self, r):  # r: [E]
        x = torch.clamp(1 - r / self.r_cut, min=0.0)        # [E]
        env = x**2 * (3 - 2*x)                              # C2 smooth
        rbf = torch.exp(-self.gamma * (r.unsqueeze(-1) - self.centers)**2)  # [E, K]
        return env.unsqueeze(-1) * rbf 

# class RBF(nn.Module):
#     def __init__(self, centers, gamma, cutoff):
#         super().__init__()
#         self.register_buffer('c', torch.tensor(centers).float())
#         self.gamma = gamma
#         self.cutoff = cutoff
#     def forward(self, r):
#         # smooth envelope for invariance + stability
#         x = torch.clamp(1 - (r / self.cutoff), min=0.0)
#         env = x**2 * (3 - 2*x)  # C2 continuous
#         rbf = torch.exp(-self.gamma * (r.unsqueeze(-1) - self.c)**2)
#         return env.unsqueeze(-1) * rbf  # [E, 1] * [E, K] -> [E, K]

class RBF_Naive(nn.Module):
    def __init__(self, centers, gamma):
        super().__init__()
        self.register_buffer('c', torch.tensor(centers).float())
        self.gamma = gamma
    def forward(self, r):
        # r: [E] distances
        return torch.exp(-self.gamma * (r.unsqueeze(-1) - self.c)**2)  # [E, K]

class PairEnergyReadout(nn.Module):
    def __init__(self, hdim, rbf_centers, rbf_gamma=10.0):
        super().__init__()
        self.rbf = RBF(rbf_centers, rbf_gamma)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hdim + len(rbf_centers), 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    def forward(self, h0, pos, edge_index, ligand_mask=None, protein_mask=None):
        # h0: [N, F0] scalar channels
        src, dst = edge_index  # [E]
        rij = pos[src] - pos[dst]       # [E, 3]
        r = torch.linalg.norm(rij, dim=-1)  # [E]
        phi_r = self.rbf(r)                 # [E, K]

        h_ij = torch.cat([h0[src], h0[dst], phi_r], dim=-1)  # [E, 2F0+K]
        e = self.edge_mlp(h_ij).squeeze(-1)                  # [E]

        # (Optional) keep only ligand–protein cross edges for the sum:
        if ligand_mask is not None and protein_mask is not None:
            keep = (ligand_mask[src] & protein_mask[dst]) | (protein_mask[src] & ligand_mask[dst])
            e = e[keep]

        y = e.sum()  # invariant global scalar
        return y

class InterfaceReadout(nn.Module):
    def __init__(self, hdim, rbf_centers, rbf_gamma=10.0, cutoff=8.0, hidden=128,
                 use_attention=False, normalize=False):
        super().__init__()
        self.rbf = RBF(rbf_centers, rbf_gamma, cutoff)
        in_dim = 2*hdim + len(rbf_centers)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        self.use_attention = use_attention
        self.normalize = normalize
        if use_attention:
            self.att = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
            )
        self.cutoff = cutoff

    @torch.no_grad()
    def _pair_mask(self, pos, src, dst):
        rij = pos[src] - pos[dst]
        r = torch.linalg.norm(rij, dim=-1)
        return r, (r < self.cutoff)

    def forward(self, h_scalar, pos, edge_index, lig_mask, prot_mask, elem_pair=None, charge_pair=None):
        """
        h_scalar: [N, F0] final scalar channels from equivariant backbone
        pos: [N, 3]
        edge_index: [2, E] dense candidate edges (e.g., all pairs or kNN)
        lig_mask, prot_mask: [N] booleans
        elem_pair, charge_pair (optional): [E, D] additional scalar invariants
        """
        src, dst = edge_index
        # keep only ligand–protein edges
        lp = (lig_mask[src] & prot_mask[dst]) | (prot_mask[src] & lig_mask[dst])
        src, dst = src[lp], dst[lp]

        rij = pos[src] - pos[dst]
        r = torch.linalg.norm(rij, dim=-1)                      # [E]
        keep = r < self.cutoff
        src, dst, r = src[keep], dst[keep], r[keep]

        phi_r = self.rbf(r)                                     # [E, K]
        feats = [h_scalar[src], h_scalar[dst], phi_r]
        if elem_pair is not None: 
            feats.append(elem_pair[lp][keep])
        if charge_pair is not None: 
            feats.append(charge_pair[lp][keep])
        x_ij = torch.cat(feats, dim=-1)                         # [E, 2F0+K(+extras)]

        s_ij = self.edge_mlp(x_ij).squeeze(-1)                  # [E]

        if self.use_attention:
            a_ij = self.att(x_ij).squeeze(-1)                   # [E]
            w = torch.softmax(a_ij, dim=0)                      # invariant attention
            y = (w * s_ij).sum()
        else:
            y = s_ij.sum()

        if self.normalize:
            y = y / (s_ij.numel() + 1e-6)                       # optional for non-extensive targets

        return y


class PairBranch(nn.Module):
    """Invariant sum over pairs for a given edge set."""
    def __init__(self, hdim, rbf_centers, cutoff, gamma=10.0, hidden=128, extras_dim=0, normalize=False):
        super().__init__()
        self.rbf = RBF(rbf_centers, gamma, cutoff)
        self.normalize = normalize
        in_dim = 2*hdim + len(rbf_centers) + extras_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        self.cutoff = cutoff

    def forward(self, h0, pos, edge_index, extras=None):
        # h0: [N, F0] scalar channels from equivariant backbone
        # pos: [N, 3], edge_index: [2, E]
        src, dst = edge_index
        rij = pos[src] - pos[dst]
        r = torch.linalg.norm(rij, dim=-1)             # [E]
        keep = r < self.cutoff
        if keep.sum() == 0:
            return h0.new_zeros(())
        src, dst, r = src[keep], dst[keep], r[keep]
        phi_r = self.rbf(r)                            # [E, K]
        feats = [h0[src], h0[dst], phi_r]
        if extras is not None:
            feats.append(extras[keep])                 # e.g., elem/charge pairs (invariant)
        x_ij = torch.cat(feats, dim=-1)                # [E, 2F0+K+extras]
        s = self.mlp(x_ij).squeeze(-1)                 # [E]
        return (s.mean() if self.normalize else s.sum())

class MultiBranchInvariantReadout(nn.Module):
    def __init__(self, hdim, rbf_centers=tuple(torch.linspace(0, 8.0, 64).tolist()),
                 cut_cross=8.0, cut_lig=5.0, hidden=128,
                 extras_dim_cross=0, extras_dim_lig=0,
                 include_pocket_summary=False, normalize_cross=False, normalize_lig=True):
        super().__init__()
        self.cross = PairBranch(hdim, rbf_centers, cut_cross, hidden=hidden,
                                extras_dim=extras_dim_cross, normalize=normalize_cross)
        self.lig = PairBranch(hdim, rbf_centers, cut_lig, hidden=hidden,
                              extras_dim=extras_dim_lig, normalize=normalize_lig)
        self.include_pocket = include_pocket_summary
        if include_pocket_summary:
            self.pocket_head = nn.Sequential(nn.Linear(hdim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        comb_in = 2 + int(self.include_pocket)
        self.comb = nn.Sequential(
            nn.Linear(comb_in, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, h0, pos, edge_index_allpairs, lig_mask, prot_mask,
                extras_cross=None, extras_lig=None):
        src, dst = edge_index_allpairs

        # cross edges L-P (both directions allowed)
        lp = (lig_mask[src] & prot_mask[dst]) | (prot_mask[src] & lig_mask[dst])
        edge_cross = torch.stack([src[lp], dst[lp]], dim=0)

        # intra-ligand edges L-L (i<j to avoid double)
        ll = lig_mask[src] & lig_mask[dst] & (src < dst)
        edge_lig = torch.stack([src[ll], dst[ll]], dim=0)

        z_cross = self.cross(h0, pos, edge_cross, extras_cross)
        z_lig   = self.lig(h0, pos, edge_lig, extras_lig)

        parts = [z_cross, z_lig]
        if self.include_pocket:
            z_pocket = self.pocket_head(h0[prot_mask]).sum()
            parts.append(z_pocket)

        z = torch.stack(parts, dim=-1)      # [3] or [2]
        y = self.comb(z)                    # scalar prediction
        return y.squeeze(-1)





class InvariantScalarReadout(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.att = nn.Linear(in_dim, 1)   # attention on scalars (invariant)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, h0, mask=None):
        # h0: [N, F0] scalar channels from the equivariant backbone
        a = self.att(h0).squeeze(-1)      # [N]
        if mask is not None:              # mask padded atoms
            a = a.masked_fill(~mask, float('-inf'))
        w = torch.softmax(a, dim=0)       # invariant attention weights
        z = (w.unsqueeze(-1) * h0).sum(dim=0)  # [F0]
        return self.mlp(z).squeeze(-1)         # scalar


class ChemPairFeatures(nn.Module):
    def __init__(self, num_elements=10, emb_dim=32):
        super().__init__()
        # symmetric element-pair embedding table
        max_pair_id = num_elements*(num_elements+1)//2
        self.elem_pair_emb = nn.Embedding(max_pair_id, emb_dim)

    def forward(self, Z_i, Z_j, charges=None, donor=None, acceptor=None):
        pair_id = torch.where(Z_i >= Z_j,
                              (Z_i*(Z_i+1))//2 + Z_j,
                              (Z_j*(Z_j+1))//2 + Z_i)
        e_emb = self.elem_pair_emb(pair_id)

        feats = [e_emb]
        if charges is not None:
            q_i, q_j = charges
            feats += [q_i.unsqueeze(-1), q_j.unsqueeze(-1),
                      (q_i*q_j).unsqueeze(-1), torch.abs(q_i-q_j).unsqueeze(-1)]
        if donor is not None and acceptor is not None:
            feats += [(donor*acceptor).unsqueeze(-1),
                      (acceptor*donor).unsqueeze(-1)]
        return torch.cat(feats, dim=-1)  # [E, D_chem]
    


# class S_Predictor(nn.Module()):
#     def __init__(self, egnn: nn.Module):
#         self.egnn = egnn
    
#     def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues):
#         x_atoms = xh_atoms[:, :self.n_dims].clone()
#         h_atoms = xh_atoms[:, self.n_dims:].clone()

#         x_residues = xh_residues[:, :self.n_dims].clone()
#         h_residues = xh_residues[:, self.n_dims:].clone()

#         # embed atom features and residue features in a shared space
#         h_atoms = self.atom_encoder(h_atoms)
#         h_residues = self.residue_encoder(h_residues)

#         # combine the two node types
#         x = torch.cat((x_atoms, x_residues), dim=0)
#         h = torch.cat((h_atoms, h_residues), dim=0)
#         mask = torch.cat([mask_atoms, mask_residues])
