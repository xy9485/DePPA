import numpy as np
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from analysis.SA_Score.sascorer import calculateScore

from analysis.molecule_builder import build_molecule
from copy import deepcopy


class CategoricalDistribution:
    EPS = 1e-10

    def __init__(self, histogram_dict, mapping):
        histogram = np.zeros(len(mapping))
        for k, v in histogram_dict.items():
            histogram[mapping[k]] = v

        # Normalize histogram
        self.p = histogram / histogram.sum()
        self.mapping = deepcopy(mapping)

    def kl_divergence(self, other_sample):
        sample_histogram = np.zeros(len(self.mapping))
        for x in other_sample:
            # sample_histogram[self.mapping[x]] += 1
            sample_histogram[x] += 1

        # Normalize
        q = sample_histogram / sample_histogram.sum()

        return -np.sum(self.p * np.log(q / self.p + self.EPS))


def rdmol_to_smiles(rdmol):
    mol = Chem.Mol(rdmol)
    Chem.RemoveStereochemistry(mol)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, dataset_smiles_list=None,
                 connectivity_thresh=1.0):
        self.atom_decoder = dataset_info['atom_decoder']
        if dataset_smiles_list is not None:
            dataset_smiles_list = set(dataset_smiles_list)
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info
        self.connectivity_thresh = connectivity_thresh

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        if len(generated) < 1:
            return [], 0.0

        valid = []
        for mol in generated:
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                continue

            valid.append(mol)

        return valid, len(valid) / len(generated)

    def compute_connectivity(self, valid):
        """ Consider molecule connected if its largest fragment contains at
        least x% of all atoms, where x is determined by
        self.connectivity_thresh (defaults to 100%). """
        if len(valid) < 1:
            return [], 0.0

        connected = []
        connected_smiles = []
        for mol in valid:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            largest_mol = \
                max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            if largest_mol.GetNumAtoms() / mol.GetNumAtoms() >= self.connectivity_thresh:
                smiles = rdmol_to_smiles(largest_mol)
                if smiles is not None:
                    connected_smiles.append(smiles)
                    connected.append(largest_mol)

        return connected, len(connected_smiles) / len(valid), connected_smiles

    def compute_uniqueness(self, connected):
        """ valid: list of SMILES strings."""
        if len(connected) < 1 or self.dataset_smiles_list is None:
            return [], 0.0

        return list(set(connected)), len(set(connected)) / len(connected)

    def compute_novelty(self, unique):
        if len(unique) < 1:
            return [], 0.0

        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate_rdmols(self, rdmols):
        valid, validity = self.compute_validity(rdmols)
        print(f"Validity over {len(rdmols)} molecules: {validity * 100 :.2f}%")

        connected, connectivity, connected_smiles = \
            self.compute_connectivity(valid)
        print(f"Connectivity over {len(valid)} valid molecules: "
              f"{connectivity * 100 :.2f}%")

        unique, uniqueness = self.compute_uniqueness(connected_smiles)
        print(f"Uniqueness over {len(connected)} connected molecules: "
              f"{uniqueness * 100 :.2f}%")

        _, novelty = self.compute_novelty(unique)
        print(f"Novelty over {len(unique)} unique connected molecules: "
              f"{novelty * 100 :.2f}%")

        return [validity, connectivity, uniqueness, novelty], [valid, connected]

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """

        rdmols = [build_molecule(*graph, self.dataset_info)
                  for graph in generated]
        return self.evaluate_rdmols(rdmols)


class MoleculeProperties:

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = calculateScore(rdmol)
        return round((10 - sa) / 9, 2)  # from pocket2mol

    @staticmethod
    def calculate_logp(rdmol):
        return Crippen.MolLogP(rdmol)

    @staticmethod
    def calculate_docking_score(ligand_mol, receptor_pdbqt_file, center_xyz, use_meeko=False, size=20, exhaustiveness=16, score_only=False):
        from openbabel import pybel
        import os
        import tempfile
        assert ligand_mol.GetNumConformers() > 0
        # mol = Chem.AddHs(ligand_mol)
        # 3D coordinates + quick minimization (ETKDG + MMFF or UFF)
        # Chem.AllChem.EmbedMolecule(mol, Chem.AllChem.ETKDGv3())
        # Chem.AllChem.MMFFOptimizeMolecule(mol)  # or Chem.AllChem.UFFOptimizeMolecule(mol)
        # center box at ligand's center of mass

        # cx, cy, cz = ligand_mol.GetConformer().GetPositions().mean(0)
        cx, cy, cz = center_xyz
        try:
            # Use a unique temp directory per call to avoid collisions when
            # called concurrently from multiple threads/processes.
            with tempfile.TemporaryDirectory(prefix="qvina_temp_") as tmpdir:
                lig_pdbqt = os.path.join(tmpdir, "ligand.pdbqt")

                if use_meeko:
                    from meeko import MoleculePreparation, PDBQTWriterLegacy

                    # Prepare for docking
                    preparer = MoleculePreparation()
                    prepared_list = preparer.prepare(ligand_mol)
                    assert len(prepared_list) == 1
                    prepared = prepared_list[0]
                    writer = PDBQTWriterLegacy()
                    pdbqt_string, success, error_msg = writer.write_string(prepared)
                    assert success, error_msg
                    # Write ligand to a unique temp path
                    with open(lig_pdbqt, "w") as f:
                        f.write(pdbqt_string)
                else:
                    # RDKit -> molblock (SDF text)
                    molblock = Chem.MolToMolBlock(ligand_mol, kekulize=False)
                    # pybel read and write PDBQT
                    obmol = pybel.readstring("sdf", molblock)
                    obmol.write("pdbqt", lig_pdbqt, overwrite=True)

                # run QuickVina 2 (optionally in score-only mode)
                cmd = (
                    f'qvina2 --receptor "{receptor_pdbqt_file}" '
                    f'--ligand "{lig_pdbqt}" '
                    f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
                    f'--size_x {size} --size_y {size} --size_z {size} '
                    f'--exhaustiveness {exhaustiveness}'
                )
                if score_only:
                    cmd += ' --score_only'
                out = os.popen(cmd).read()
                # write out into a log file named docking_log.txt in working directory, uncomment for debugging
                # with open("docking_log.txt", "a") as log_file:
                #     log_file.write(out + "\n")

            # Parse output: docking table (normal mode) or Affinity line (score-only)
            if '-----+------------+----------+----------' not in out:
                # Try to parse score-only style output: "Affinity: <value> (kcal/mol)"
                try:
                    import re
                    m = re.search(r'Affinity:\s*([-+]?\d*\.?\d+)', out)
                    if m:
                        score = float(m.group(1))
                        return -score
                    else:
                        raise ValueError("No docking score found in output.")
                except Exception:
                    pass
                # score = np.nan
                # return -score

            out_split = out.splitlines()
            best_idx = out_split.index('-----+------------+----------+----------') + 1
            best_line = out_split[best_idx].split()
            assert best_line[0] == '1'
            score=float(best_line[1])
            return -score

        except Exception as e:
            print(f"Error calculating docking score: {e}")
            score = np.nan
            return -score

    @staticmethod
    def calculate_lipinski(rdmol):
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

    @classmethod
    def calculate_diversity(cls, pocket_mols):
        if len(pocket_mols) < 2:
            return 0.0

        div = 0
        total = 0
        for i in range(len(pocket_mols)):
            for j in range(i + 1, len(pocket_mols)):
                div += 1 - cls.similarity(pocket_mols[i], pocket_mols[j])
                total += 1
        return div / total

    @staticmethod
    def similarity(mol_a, mol_b):
        # fp1 = AllChem.GetMorganFingerprintAsBitVect(
        #     mol_a, 2, nBits=2048, useChirality=False)
        # fp2 = AllChem.GetMorganFingerprintAsBitVect(
        #     mol_b, 2, nBits=2048, useChirality=False)
        fp1 = Chem.RDKFingerprint(mol_a)
        fp2 = Chem.RDKFingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    @staticmethod
    def calculate_diversity_morgan(rdmols, radius=2, n_bits=2048, use_chirality=False):
        """
        Compute diversity as the average pairwise Tanimoto similarity over
        Morgan fingerprints among a set of RDKit molecules.

        Args:
            rdmols: list of RDKit Mol objects.
            radius: Morgan fingerprint radius (default 2).
            n_bits: Number of bits for Morgan fingerprints (default 2048).
            use_chirality: Whether to use chiral information (default False).

        Returns:
            float: Mean pairwise Tanimoto similarity across all unique pairs.
                   Returns 0.0 if fewer than 2 valid molecules are provided.
        """
        if rdmols is None or len(rdmols) < 2:
            return 0.0

        # Local import to avoid changing global imports
        from rdkit.Chem import AllChem

        # Build fingerprints; skip None entries defensively
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits, useChirality=use_chirality)
            for m in rdmols if m is not None
        ]

        if len(fps) < 2:
            return 0.0

        total_pairs = 0
        sim_sum = 0.0
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim_sum += DataStructs.TanimotoSimilarity(fps[i], fps[j])
                total_pairs += 1

        return sim_sum / total_pairs if total_pairs > 0 else 0.0

    def evaluate(self, pocket_rdmols):
        """
        Run full evaluation
        Args:
            pocket_rdmols: list of lists, the inner list contains all RDKit
                molecules generated for a pocket
        Returns:
            QED, SA, LogP, Lipinski (per molecule), and Diversity (per pocket)
        """

        for pocket in pocket_rdmols:
            for mol in pocket:
                Chem.SanitizeMol(mol)
                assert mol is not None, "only evaluate valid molecules"

        all_qed = []
        all_sa = []
        all_logp = []
        all_lipinski = []
        per_pocket_diversity = []
        for pocket in tqdm(pocket_rdmols):
            all_qed.append([self.calculate_qed(mol) for mol in pocket])
            all_sa.append([self.calculate_sa(mol) for mol in pocket])
            all_logp.append([self.calculate_logp(mol) for mol in pocket])
            all_lipinski.append([self.calculate_lipinski(mol) for mol in pocket])
            per_pocket_diversity.append(self.calculate_diversity(pocket))

        print(f"{sum([len(p) for p in pocket_rdmols])} molecules from "
              f"{len(pocket_rdmols)} pockets evaluated.")

        qed_flattened = [x for px in all_qed for x in px]
        print(f"QED: {np.mean(qed_flattened):.3f} \pm {np.std(qed_flattened):.2f}")

        sa_flattened = [x for px in all_sa for x in px]
        print(f"SA: {np.mean(sa_flattened):.3f} \pm {np.std(sa_flattened):.2f}")

        logp_flattened = [x for px in all_logp for x in px]
        print(f"LogP: {np.mean(logp_flattened):.3f} \pm {np.std(logp_flattened):.2f}")

        lipinski_flattened = [x for px in all_lipinski for x in px]
        print(f"Lipinski: {np.mean(lipinski_flattened):.3f} \pm {np.std(lipinski_flattened):.2f}")

        print(f"Diversity: {np.mean(per_pocket_diversity):.3f} \pm {np.std(per_pocket_diversity):.2f}")

        return all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity

    def evaluate_mean(self, rdmols):
        """
        Run full evaluation and return mean of each property
        Args:
            rdmols: list of RDKit molecules
        Returns:
            QED, SA, LogP, Lipinski, and Diversity
        """

        if len(rdmols) < 1:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        for mol in rdmols:
            Chem.SanitizeMol(mol)
            assert mol is not None, "only evaluate valid molecules"

        qed = np.mean([self.calculate_qed(mol) for mol in rdmols])
        sa = np.mean([self.calculate_sa(mol) for mol in rdmols])
        logp = np.mean([self.calculate_logp(mol) for mol in rdmols])
        lipinski = np.mean([self.calculate_lipinski(mol) for mol in rdmols])
        diversity = self.calculate_diversity(rdmols)

        return qed, sa, logp, lipinski, diversity
