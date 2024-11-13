import os
import sys
sys.path.append('./')
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


NANO_TO_ANGSTROM = 0.1
# Constants
COORDS = 'coords_tensor.pt'
DELTA_G = 'deltaG.pt'
MASKS = 'mask_tensor.pt'
ONE_HOT = 'one_hot_encodings.pt'
PROTT5_EMBEDDINGS = 'prott5_embeddings'
VAL_RATIO = 0.2
RANDOM_SEED = 42
NANO_TO_ANGSTROM = 0.1
TM_PATH = './data/Processed_K50_dG_datasets/TM_proteins.csv'
DEBUG = False

def normalize_batch(batch, LLM_EMB = True):
    batch['one_hot'] = batch['one_hot'][:, :, :, :-1]
    batch['coords'] = batch['coords'] * NANO_TO_ANGSTROM
    if not LLM_EMB: # zero prot5 embedding
         batch['prott5'] = torch.zeros_like(batch['prott5'])
    return batch    

class AllProteinValidationDataset(Dataset):

    def __init__(self, tensor_root_dir, mutations_root_dir, train  = True, one_mut = True):
        self.tensor_root_dir = tensor_root_dir
        self.mutations_root_dir = mutations_root_dir
        self.protein_dirs = [protein for i, protein in enumerate(os.listdir(self.tensor_root_dir))]
        self.one_mut = one_mut # remove the mutations with more than one mutation
        # remove TM proteins 
        tm_proteins = pd.read_csv(TM_PATH)
        tm_proteins = tm_proteins['name'].apply(lambda x: x.split(".")[0]).unique().tolist()
        self.protein_dirs = [protein for protein in self.protein_dirs if protein not in tm_proteins]
        if DEBUG:
            self.protein_dirs = self.protein_dirs[:5]
        # # Train test split
        # self.training_protein, self.val_proteins = train_test_split(self.protein_dirs, test_size=VAL_RATIO, random_state=RANDOM_SEED)
        # if train:
        #     self.protein_dirs = self.training_protein
        # else:
        #     self.protein_dirs = self.val_proteins

    def __len__(self):
        return len(self.protein_dirs)

    def __getitem__(self, idx):
        protein_dir = os.path.join(self.tensor_root_dir, self.protein_dirs[idx])
        mutations_path = os.path.join(self.mutations_root_dir, f'{self.protein_dirs[idx]}.csv')
        mutations = pd.read_csv(mutations_path)
        mutations = mutations[~mutations['mut_type'].str.contains('ins|del')].reset_index(drop=True)
        # Load and preprocess the data for each protein
        coords_tensor = torch.load(os.path.join(protein_dir, COORDS))
        delta_g_tensor = torch.load(os.path.join(protein_dir, DELTA_G))
        mask_tensor = torch.load(os.path.join(protein_dir, MASKS))
        one_hot_tensor = torch.load(os.path.join(protein_dir, ONE_HOT))
        embedding_tensor = self.load_embedding_tensor(os.path.join(protein_dir, PROTT5_EMBEDDINGS))
        
        # remove the mutations with more than one mutation
        if self.one_mut:
            one_mut_index = mutations[~mutations['mut_type'].str.contains(':')]
            mutations = mutations.loc[one_mut_index.index]
            delta_g_tensor = delta_g_tensor[one_mut_index.index]
            one_hot_tensor = one_hot_tensor[one_mut_index.index]
            embedding_tensor = embedding_tensor[one_mut_index.index]
            
        mutations_data = {
            'name': self.protein_dirs[idx],
            'mutations': mutations['mut_type'].to_list(),
            'prott5': embedding_tensor,
            'coords': coords_tensor,
            'one_hot': one_hot_tensor,
            'delta_g': delta_g_tensor,
            'masks': mask_tensor
        }

        return mutations_data

    def load_embedding_tensor(self, embeddings_dir):
        embeddings = []
        all_embedding_files = sorted(glob.glob(os.path.join(embeddings_dir, 'prott5_embedding_*.pt')),
                                     key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
        for filename in all_embedding_files:
            if filename.endswith('.pt'):
                embedding_tensor = torch.load(filename)
                embeddings.append(embedding_tensor)
        return torch.vstack(embeddings)
    
    
if __name__ == "__main__":
    Ds = AllProteinValidationDataset('./data/Processed_K50_dG_datasets/AlphaFold_model_PDBs', './data/Processed_K50_dG_datasets/mutation_datasets')
    print(Ds[0])