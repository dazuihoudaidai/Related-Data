import os
import pickle
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import pymol
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def convert_ligand_format(data_folder, complex_id, ligand_format):
    ligand_input = os.path.join(data_folder, complex_id, f'{complex_id}_ligand.{ligand_format}')
    ligand_pdb = ligand_input.replace(f".{ligand_format}", ".pdb")
    os.system(f'obabel {ligand_input} -O {ligand_pdb} -d')
    return ligand_pdb

def load_molecule(file_path):
    molecule = Chem.MolFromPDBFile(file_path, removeHs=True)
    if molecule is None:
        return None
    return molecule

def generate_pocket_and_ligand(data_folder, complex_id, radius, ligand_format):
    complex_path = os.path.join(data_folder, complex_id)
    protein_file = os.path.join(complex_path, f"{complex_id}_protein.pdb")
    pocket_file = os.path.join(complex_path, f'Pocket_{radius}A.pdb')

    if os.path.exists(pocket_file):
        return None, None

    ligand_file = os.path.join(complex_path, f"{complex_id}_ligand.mol2")
    
    pymol.cmd.load(protein_file)
    pymol.cmd.remove('resn HOH')
    pymol.cmd.load(ligand_file)
    pymol.cmd.remove('hydrogens')
    pymol.cmd.select('Pocket', f'byres {complex_id}_ligand around {radius}')
    pymol.cmd.save(pocket_file, 'Pocket')
    pymol.cmd.delete('all')

    if ligand_format != 'pdb':
        ligand_pdb = convert_ligand_format(data_folder, complex_id, ligand_format)
    else:
        ligand_pdb = os.path.join(complex_path, f'{complex_id}_ligand.pdb')

    ligand_molecule = load_molecule(ligand_pdb)
    pocket_molecule = load_molecule(pocket_file)
    
    if ligand_molecule is None or pocket_molecule is None:
        print(f"Failed to process {complex_id}")
        return None, None

    return ligand_molecule, pocket_molecule

def process_complexes(data_folder, complexes_df, radius=5, ligand_format='mol2'):
    progress_bar = tqdm(total=len(complexes_df))

    for _, row in complexes_df.iterrows():
        complex_id = row['pdbid']
        complex_path = os.path.join(data_folder, complex_id)
        save_file = os.path.join(complex_path, f"{complex_id}_{radius}A.rdkit")

        ligand_molecule, pocket_molecule = generate_pocket_and_ligand(data_folder, complex_id, radius, ligand_format)

        if ligand_molecule and pocket_molecule:
            complex_data = (ligand_molecule, pocket_molecule)
            with open(save_file, 'wb') as file:
                pickle.dump(complex_data, file)

        progress_bar.update(1)

if __name__ == '__main__':
    radius = 5
    ligand_format = 'mol2'
    root_folder = './data'
    data_folder = os.path.join(root_folder, 'validationset')
    complexes_df = pd.read_csv(os.path.join(root_folder, 'validationset.csv'))

    process_complexes(data_folder, complexes_df, radius=radius, ligand_format=ligand_format)
