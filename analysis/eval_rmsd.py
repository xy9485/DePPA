"""
conda install spyrmsd -c conda-forge
pip install spyrmsd

"""
from spyrmsd import molecule
from spyrmsd import rmsd as spy_rmsd
import torch
import argparse
from openbabel import openbabel as ob
from openbabel import pybel
import os
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import time

ob.obErrorLog.SetOutputLevel(0)

def get_symmetry_rmsd(mol, ref):
    # with time_limit(10):
    try:
        mol = molecule.Molecule.from_rdkit(mol)
        ref = molecule.Molecule.from_rdkit(ref)
        coords_ref = ref.coordinates
        anum_ref = ref.atomicnums
        adj_ref = ref.adjacency_matrix
        coords = mol.coordinates
        anum = mol.atomicnums
        adj = mol.adjacency_matrix
        rmsd_val = spy_rmsd.symmrmsd(
            coords_ref,
            coords,
            anum_ref,
            anum,
            adj_ref,
            adj,
        )
        return rmsd_val
    except Exception as e:
        print(e)
        return np.nan

def get_rmsd(gen_mol, dock_mol):
    gen_pose = gen_mol.GetConformer().GetPositions()
    dock_pose = dock_mol.GetConformer().GetPositions()
    return np.sqrt(np.sum((gen_pose - dock_pose)**2))

def get_pdbqt_mol(pdbqt_block: str) -> Chem.Mol:
    """Convert pdbqt block to rdkit mol by converting with openbabel"""

    # read pdbqt file from autodock
    mol = ob.OBMol()
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "pdb")
    obConversion.ReadString(mol, pdbqt_block)

    # convert to RDKIT
    mol = Chem.MolFromPDBBlock(obConversion.WriteString(mol))

    return mol

def get_mol_from_pdbqt_file(pdbqt_path: str) -> Chem.Mol:
    """
    Read a .pdbqt file and convert it to an RDKit Mol via Open Babel.
    Converts PDBQT $\rightarrow$ PDB $\rightarrow$ RDMol.
    """
    obmol = ob.OBMol()
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "pdb")

    if not obConversion.ReadFile(obmol, pdbqt_path):
        raise ValueError(f"Failed to read PDBQT file: {pdbqt_path}")

    pdb_block = obConversion.WriteString(obmol)
    # mol = Chem.MolFromPDBBlock(pdb_block)
    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False)
    # mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
    return mol

def pdbqt_to_rdmol_openbabel(pdbqt_file):
    """
    Uses OpenBabel to read a PDBQT and return the best pose as an RDMol.
    Converts PDBQT $\rightarrow$ MDL Mol (SDF) $\rightarrow$ RDMol.
    """
    # Read the file using pybel (pdbqt format)
    # pybel.readfile returns an iterator of molecules
    mols = list(pybel.readfile("pdbqt", pdbqt_file))
    
    if not mols:
        return None
    
    # The first molecule in the list is the best pose (Pose 1)
    best_pose = mols[0]
    
    # Convert OpenBabel molecule to an RDKit molecule
    # We use the 'mol' (MDL Molfile) format as a bridge to preserve bond orders
    mol_block = best_pose.write("mol")
    # rdmol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    rdmol = Chem.MolFromMolBlock(mol_block)
    return rdmol

def pdbqt_block_to_rdmol_openbabel(pdbqt_block: str):
    """
    OpenBabel conversion of an in-memory PDBQT block to an RDKit Mol.
    Mirrors pdbqt_to_rdmol_openbabel but accepts a PDBQT string instead of a file path.
    """
    obmol = ob.OBMol()
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "mol")

    if not obConversion.ReadString(obmol, pdbqt_block):
        return None

    mol_block = obConversion.WriteString(obmol)
    rdmol = Chem.MolFromMolBlock(mol_block)
    return rdmol

def pdbqt_block_to_rdmol_pybel(pdbqt_block: str):
    """
    OpenBabel conversion of an in-memory PDBQT block to an RDKit Mol.
    Mirrors pdbqt_to_rdmol_openbabel but accepts a PDBQT string instead of a file path.
    """
    try:
        # pybel.readstring can parse an in-memory PDBQT string similarly to readfile on disk
        obmol = pybel.readstring("pdbqt", pdbqt_block)
    except Exception:
        return None

    if obmol is None:
        return None

    mol_block = obmol.write("mol")
    rdmol = Chem.MolFromMolBlock(mol_block)
    return rdmol

def get_rmsd_between_mols(mol, docked_mol):
    if mol is None or docked_mol is None:
        return np.nan
    mol = Chem.RemoveAllHs(mol)
    docked_mol = Chem.RemoveAllHs(docked_mol)
    try:
        rmsd_val = get_symmetry_rmsd(mol, docked_mol)
    except Exception as e:
        print(e, 'not rescued')
        rmsd_val = get_rmsd(mol, docked_mol)
    return rmsd_val


def get_rmsd_between_mol_pdbqt(mol, docked_pdbqt):
    try:
        docked_mol = get_pdbqt_mol(docked_pdbqt)
        return get_rmsd_between_mols(mol, docked_mol)
    except Exception as e:
        print("Exception in get_rmsd_between_mol_pdbqt:", e)
        return np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol_dir", type=str, default="molecules_sota_docked")

    args = parser.parse_args()

    if args.mol_dir.endswith(".pt"):
        results = torch.load(args.mol_dir)
    else:
        results = torch.load(os.path.join(args.mol_dir, "eval_all.pt"))
    rmsds = []
    proxy = 0
    for res in tqdm(results, desc="Calculating RMSD"):
        if isinstance(res, dict):
            res = [res]
        assert isinstance(res, list)
        for r in res:
            mol = r["mol"]
            docked_pdbqt = r["vina"]['dock'][0]['pose']
            docked_mol = get_pdbqt_mol(docked_pdbqt)
            if mol is None or docked_mol is None:
                continue
            mol = Chem.RemoveAllHs(mol)
            docked_mol = Chem.RemoveAllHs(docked_mol)
            try:
                rmsd_val = get_symmetry_rmsd(docked_mol, mol)
                rmsds.append(rmsd_val)
            except Exception as e:
                continue
                print(e)
                try:
                    rmsd_val = get_rmsd(mol, docked_mol)
                    proxy += 1
                    rmsds.append(rmsd_val)
                except Exception as e:
                    print(e, 'not rescued')
                    continue

    print(args.mol_dir, np.mean(rmsds))
    print(np.quantile(rmsds, 0.25), np.median(rmsds), np.quantile(rmsds, 0.75))

    # calc ratio of rmsd < 2
    print(np.mean(np.array(rmsds) < 2), len(rmsds), proxy)
    np.save(f"{os.path.basename(args.mol_dir)}_rmsds.npy", rmsds)