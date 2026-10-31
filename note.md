# DiffSBDD - Working Notes

This document records small, important details discovered while working with this repo. Entries are concise, actionable, and include a quick way to verify when applicable.

---

## 2025-10-19 • Checkpoint norm_values interpretation

Context:
- Loading a model via `LigandPocketDDPM.load_from_checkpoint(..., map_location=device)` yields `model.norm_values = [1, 4]`.
- Interpretation: `1` applies to `x` (coordinates) and `4` applies to `h` (features/one-hot channels).
- Observation source: using `example/3rfm.pdb` as the pocket file.

Details:
- `model.norm_values = [1, 4]` means coordinate normalization scale is 1, and the feature normalization scale is 4.
- The value `4` corresponds to the number of unique pocket atom elements present in the specific pocket parsed from `3rfm.pdb`, even though `self.pocket_type_encoder` includes 10 possible types: `{ 'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9 }`.
  - In other words, only 4 of these element types actually appear in the selected pocket; the model’s normalization captures what is observed, not the full encoder vocabulary.

Code snippet used:
```python
model = LigandPocketDDPM.load_from_checkpoint(
    args.checkpoint, map_location=device)
print(model.norm_values)  # -> [1, 4]
```

---

## 2025-11-20 • Batch PDB → PDBQT conversion with `docking_py27.py`

Context:
- AutoDockTools’ `prepare_receptor4.py` (Python 2.7 tooling) remains the most reliable way we have to type receptor pockets, and `analysis/docking_py27.py` batch-wraps it.
- The `adt` conda env on the cluster already bundles Python 2.7 + MGLTools; activating it ensures `prepare_receptor4.py` is on `$PATH`.

Details / Steps:
1. Activate the AutoDockTools environment:
  ```bash
  conda activate adt
  ```
2. Run the helper with input/output directories and dataset flag (`crossdocked` or `bindingmoad`):
  ```bash
  python analysis/docking_py27.py <pdb_dir> <pdbqt_dir> <dataset>
  ```
  - `pdb_dir`: folder containing source `.pdb` pockets.
  - `pdbqt_dir`: destination folder (create it first if needed).
  - `dataset`: selects ADT options. `crossdocked` uses defaults; `bindingmoad` adds `-A checkhydrogens -e` to better preserve protonation info.
3. The script loops over every `.pdb` file, writes `<name>.pdbqt` next door, and skips outputs that already exist.

Implications / Gotchas:
- `prepare_receptor4.py` is single-threaded and sensitive to missing hydrogens; CA-only pockets still convert but electrostatics are minimal.
- Keep the `adt` env active for the entire run since the script shells out once per file.
- For very large folders, split the directory or wrap the command in GNU Parallel manually.

Notes:
- You can call the script from Python 3 if `prepare_receptor4.py` is exposed on `$PATH`, but the `adt` environment is the cleanest setup.
- For ligand prep, use `analysis/docking.py` (Python 3 + Meeko); this entry is strictly for receptor pocket conversions.

---

## 2025-11-20 • Ligand docking CLI (`analysis/docking.py`)

Context:
- `analysis/docking.py` runs ligand docking (QVina via Python 3 + Meeko) over a dataset split and expects directory-structured inputs.

Details / Invocation:
- Core arguments:
  - `--pdbqt_dir`: receptor pocket `.pdbqt` directory.
  - `--sdf_dir`: ligand `.sdf` directory.
  - `--dataset`: dataset key (e.g., `crossdock`, `bindingmoad`).
  - `--out_dir`: docking output directory (must exist or be creatable), e.g. `/home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test_qvina_out`.
- Example:
  ```bash
  python analysis/docking.py \
    --pdbqt_dir <receptor_pdbqt_dir> \
    --sdf_dir <ligand_sdf_dir> \
    --dataset crossdock \
    --out_dir /home/xue/repos/DiffSBDD/datasets2/processed_crossdock_noH_full_temp/test_qvina_out
  ```

Implications:
- When conditioning on different pockets, `h` normalization may vary if the set of observed element types changes.
- If downstream components assume a fixed feature scale tied to the encoder size (10), prefer using `len(self.pocket_type_encoder)` explicitly rather than `norm_values[1]`.
- For reproducibility across pockets, consider standardizing `norm_values` to a fixed convention during training and inference (documented in configs).

Quick verification:
```python
# Count unique pocket atom elements present in example/3rfm.pdb
from Bio.PDB import PDBParser
pdb_struct = PDBParser(QUIET=True).get_structure('', 'example/3rfm.pdb')[0]
# Adjust selection to match your pocket definition (CA-only vs all heavy atoms)
elems = [a.element.capitalize()
         for res in pdb_struct.get_residues()
         for a in res.get_atoms() if a.element != 'H']
print(sorted(set(elems)), len(set(elems)))  # should be 4 for this pocket selection
```

Notes:
- The exact pocket selection in the pipeline (CA-only vs all heavy atoms) impacts which elements are counted.
- If `virtual_nodes=True`, normalization/encodings may be affected by the virtual token; keep that in mind when interpreting `h`.

---

## How we organize this file
- Each entry starts with a date and a short title.
- Format: Context → Details → Code snippet → Implications → Quick verification → Notes.
- Keep entries minimal but self-contained. Link to files (e.g., `lightning_modules.py`) when relevant.

Suggested future entries:
- Docking pipeline quirks (Meeko vs OpenBabel, PDBQT typing).
- Stereo handling (axial/atropisomer warnings and fixes).
- Hydrogen addition and sanitization pitfalls (RDKit UpdatePropertyCache, AddHs).
- Pocket definition via `--ref_ligand` and centroid computation.
 - Investigate incorporating Fourier features (per VDM paper) for conditioning/positional encodings.

---

## 2025-10-20 • DDIM vs VDM for accelerated sampling

Summary:
- DDIM provides a deterministic, non-Markovian sampling rule that enables aggressive timestep skipping without retraining. VDM refers to a variational diffusion modeling framework/objective; acceleration is typically achieved via ODE-based samplers or distillation.

What they are:
- DDIM (Denoising Diffusion Implicit Models): a sampling formulation derived from DDPM that removes stochasticity when eta=0 and supports sub-sampling the timestep schedule. No changes to training are required to use DDIM sampling at inference.
- VDM (Variational Diffusion Models, Kingma et al., 2021): a training objective/view that parameterizes the reverse-time process (often in continuous time). Sampling can be performed via reverse SDE or probability-flow ODE; “acceleration” usually comes from better solvers or distillation, not from VDM alone.

How to accelerate in practice:
- With DDIM:
  - Use a subset of timesteps (e.g., 50–100 steps) from the training schedule and set eta=0.0 (deterministic) or a small eta for mild stochasticity.
  - Choose a stride or a custom schedule (e.g., quadratic/cosine in t) that matches the training noise schedule as closely as possible.
- With VDM-style/continuous-time samplers:
  - Use ODE solvers (Euler/Heun) or specialized diffusion ODE solvers (e.g., DPM-Solver families) with 20–50 steps.
  - For very small step counts, consider progressive distillation (student matching teacher trajectories) to preserve quality.

Trade-offs to consider (molecular generation):
- DDIM fewer steps often preserves validity/connectivity better than naïve DDPM skip, but can reduce diversity and exploration (important if you rely on stochasticity for pose/chemotype variety).
- ODE-based solvers can reach high quality with fewer steps, but need careful integration with the trained noise schedule and may require additional implementation; they can also be more sensitive to step sizes on geometric data (3D coordinates).

Recommended starting points in this repo context:
- If your model was trained with ~1000 diffusion steps:
  - DDIM: try 50–100 steps, eta=0.0 (deterministic) for maximal speed; raise eta slightly (0.05–0.2) if diversity collapses.
  - ODE-based (if available): try 20–50 steps with a 2nd-order solver (Heun/DPM-Solver-2), matching the training schedule (cosine/linear). Monitor validity/connectivity and docking throughput.
- Always re-check: validity, connectivity, fragmenting, and docking success rate; aggressive acceleration can degrade pocket fit and stereochemistry.

Notes for RL/docking workflows:
- Deterministic DDIM improves wall-clock determinism and throughput, which stabilizes reward estimates, but may limit exploration. Consider mixing a few stochastic runs (eta>0) for exploration episodes.
- Keep normalization (`norm_values`) consistent across accelerated settings; large mismatches can affect stability when step counts change.

References and pointers:
- Song et al., “Denoising Diffusion Implicit Models (DDIM)” (2020).
- Kingma et al., “Variational Diffusion Models” (2021).
- Internal discussion link provided by user: https://chatgpt.com/c/68f61375-6c94-832e-ad15-319b2b77fadf

## Rotation equivariance of the initial mean and conditional consistency

In `conditional_model.py` (e.g., in `sample_given_pocket`), the initialization
of the ligand mean position uses the per-graph center of mass of the pocket:

```
mu_lig_x = scatter_mean(pocket['x'], pocket['mask'], dim=0)
```

This choice is rotation equivariant. If the pocket coordinates undergo a rigid
rotation $R$, and the center of mass transforms as:

$$
\mu(R\,x^P) = R\,\mu(x^P).
$$

Consequently, the mean used to initialize (or condition) the ligand distribution
obeys $\mu(R\,x^P)=R\,\mu(x^P)$. This is crucial to ensure the conditional
distribution respects rotations, i.e.,

$$
p\big(R\,x_T^L\,\big|\,R\,x^P\big) = p\big(x_T^L\,\big|\,x^P\big),
$$

so sampling and likelihoods are consistent under global rotations of the
ligand–pocket system. Practically, this preserves SE(3) coherence of the model
and avoids orientation-dependent biases introduced by the initialization.
