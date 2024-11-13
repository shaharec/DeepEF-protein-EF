# fine tuning the existing model with mega-scale data
# Path: Megascale-fineTuning/train.py

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
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from model.hydro_net import PEM
from model.model_cfg import CFG
from Utils.train_utils import get_graph, get_unfolded_graph, load_checkpoint
import wandb
from tqdm import tqdm
from sklearn.model_selection import KFold

import pandas as pd

# Constants
COORDS = 'coords_tensor.pt'
DELTA_G = 'deltaG.pt'
MASKS = 'mask_tensor.pt'
ONE_HOT = 'one_hot_encodings.pt'
PROTT5_EMBEDDINGS = 'prott5_embeddings'
VAL_RATIO = 0.2
RANDOM_SEED = 42
NANO_TO_ANGSTROM = 0.1
DEBUG = False
EPOCHS = 10 if not DEBUG else 1
FREEZE_LAYERS = True
CRITERION = "L1"
MODEL_PATH = './Megascale-fineTuning/models'
MINI_BATCH_SIZE = 32
DEVICE = 'cuda'# if torch.cuda.is_available() else 'cpu'
TRAINED_MODEL_PATH = "./res/trianed_models-2cycle_drop/25_final_model.pt"
BASE_MODEL_NAME = TRAINED_MODEL_PATH.split('/')[-2]
MODEL_NAME  = BASE_MODEL_NAME
PRETRAINED = True
TM_PATH = "./data/ThermoMPNN/mega_test.csv"
LR = 1e-4
DROP_OUT = 0.2
REG_LAMBDA = 0.01

# config wandb
config = {
    'coords': COORDS,
    'delta_g': DELTA_G,
    'masks': MASKS,
    'one_hot': ONE_HOT,
    'prott5_embeddings': PROTT5_EMBEDDINGS,
    'val_ratio': VAL_RATIO,
    'random_seed': RANDOM_SEED,
    'nano_to_angstrom': NANO_TO_ANGSTROM,
    'debug': DEBUG,
    'epochs': EPOCHS,
    'freeze_layers': FREEZE_LAYERS,
    'model_path': MODEL_PATH,
    'model_name': MODEL_NAME,
    'mini_batch_size': MINI_BATCH_SIZE,
    'device': DEVICE,
    'trained_model_path': TRAINED_MODEL_PATH,
    'pretrained': PRETRAINED,
    'lr': LR,
    'dropout': DROP_OUT,
    'reg_lambda': REG_LAMBDA
}

if not os.path.exists(os.path.join(MODEL_PATH, MODEL_NAME)):
    os.makedirs(os.path.join(MODEL_PATH, MODEL_NAME))


def wandb_log(log_dict,run = None):
    if not DEBUG:
        if run is not None:
            run.log(log_dict)
        else:
            wandb.log(log_dict)

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
        coords_tensor = torch.load(os.path.join(protein_dir, COORDS),weights_only=True)
        delta_g_tensor = torch.load(os.path.join(protein_dir, DELTA_G),weights_only=True)
        mask_tensor = torch.load(os.path.join(protein_dir, MASKS),weights_only=True)
        one_hot_tensor = torch.load(os.path.join(protein_dir, ONE_HOT),weights_only=True)
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
                embedding_tensor = torch.load(filename,weights_only=True)
                embeddings.append(embedding_tensor)
        return torch.vstack(embeddings)


# Evaluation class

class Evaluate():
    def __init__(self, model, train_dl, device = DEVICE, debug = DEBUG):
        self.model = model.to(device)
        self.train_dl = train_dl
        self.device = device
        self.criterion = nn.L1Loss()
        self.model.to(self.device)
        self.mini_batch_size = MINI_BATCH_SIZE
        self.debug = debug
        

    def evaluate(self, epoch, run = None):
        """
        Evaluate the model on the trainng dataset
        args:
        run: wandb run object
        returns:
        pc_corr: float, pearson correlation
        """
        self.model.eval()
        eval_df = pd.DataFrame()
        pc_df = pd.DataFrame()
        protein_metrics_df = pd.DataFrame()
    
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.train_dl,desc=f'Evaluateing training')):
                batch = normalize_batch(batch, True)
                batch_loss = 0
                batch_idx = 1
                protein_preds = []
                protein_targets = []
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    batch_idx += 1
                    output,u_energy,f_energy = self.get_deltaG(batch, j)
                    delta_g = batch['delta_g'][0, j: j + self.mini_batch_size].to(self.device)

                    # Collect predictions and targets for this protein
                    protein_preds.extend(output.cpu().numpy())
                    protein_targets.extend(delta_g.cpu().numpy())
                    # Save mutationns
                    mutations = batch['mutations'][j: j + self.mini_batch_size]
                    mutations = [mutations[i][0] for i in range(len(mutations))]
                    # Calculate loss
                    loss = self.criterion(output,delta_g)
                    energys = torch.cat((u_energy,f_energy),dim=0)
                    energy_reg = REG_LAMBDA * (F.mse_loss(energys,torch.zeros_like(energys)))
                    loss += energy_reg
                    batch_loss += loss.item()
                    # Save protein evaluation
                    batch_dict = {
                        "Ejf": f_energy.cpu().numpy(), 
                        "Eju": u_energy.cpu().numpy(),
                        "name": batch['name'] * len(u_energy),
                        "mutations": mutations,
                        "delta_g": delta_g.cpu().numpy(), 
                        "delta_g_pred": output.cpu().numpy(),
                        "seq_len": [batch['coords'].shape[1]] * len(u_energy)
                    }
                    eval_df = pd.concat([eval_df,pd.DataFrame(batch_dict)],axis=0,ignore_index=True)
                batch_loss /= batch_idx
                
                # Calculate per-protein metrics
                protein_corr = np.corrcoef(protein_preds, protein_targets)[0,1]
                protein_mae = np.mean(np.abs(np.array(protein_preds) - np.array(protein_targets)))
                protein_rmse = np.sqrt(np.mean((np.array(protein_preds) - np.array(protein_targets))**2))

                # Store metrics in list
                protein_metrics = ({
                    'protein_name': batch['name']*len(protein_preds),
                    'num_mutations': len(protein_preds),
                    'seq_len': batch['coords'].shape[1],
                    'pearson_correlation': protein_corr,
                    'mae': protein_mae,
                    'rmse': protein_rmse,
                    'avg_loss': batch_loss / batch_idx
                })
            
                # Create DataFrame with protein metrics
                protein_metrics_df = pd.concat([protein_metrics_df,pd.DataFrame(protein_metrics)])
        return eval_df, protein_metrics_df
        
    def get_deltaG(self, batch, i):
        # move all to the same device
        one_hot_minibatch = batch['one_hot'][0, i: i + self.mini_batch_size].to(self.device)
        prott5_embedding_minibatch = batch['prott5'][0, i: i + self.mini_batch_size].to(self.device)
        batch['coords'] = batch['coords'].to(self.device)
        batch['masks'] = batch['masks'].to(self.device)
        # get the graph
        folded_graph_minibatch = torch.stack(
            [get_graph(batch['coords'].squeeze(), one_hot_minibatch[i].squeeze(), prott5_embedding_minibatch[i].squeeze(), batch['masks'].squeeze()) for i in
            range(prott5_embedding_minibatch.size(0))])
        unfolded_graph_minibatch = torch.stack(
            [get_unfolded_graph(batch['coords'].squeeze(), one_hot_minibatch[i].squeeze(), prott5_embedding_minibatch[i].squeeze(), batch['masks'].squeeze()) for i in
            range(prott5_embedding_minibatch.size(0))])

        all_graph_minibatch = torch.cat([folded_graph_minibatch, unfolded_graph_minibatch], dim=0)

        minibatch_energy = self.model(all_graph_minibatch)
        folded_energy = minibatch_energy[:minibatch_energy.size(0) // 2]
        unfolded_energy = minibatch_energy[minibatch_energy.size(0) // 2:]
        
        return unfolded_energy - folded_energy,unfolded_energy,folded_energy
        

def evaluat_model():
    # Load the model
    model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,dropout_rate = CFG.dropout_rate).to(DEVICE)
    try:
        model_dict = torch.load(TRAINED_MODEL_PATH,map_location=DEVICE,weights_only=False)
        model.load_state_dict(model_dict['model_state_dict'])
    except :
        model.load_state_dict(torch.load(TRAINED_MODEL_PATH,weights_only=True))
    model.eval()

    # Load the training dataset
    protein_train = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                              mutations_root_dir=mutations_root_dir, train=True)
    if DEBUG:
        protein_train = Subset(protein_train, list(range(5)))
    
    # Create the dataloaders
    train_dl = DataLoader(protein_train, batch_size=1, shuffle=False)

    # Evaluate the model
    evaluate = Evaluate(model, train_dl)
    eval_df, protein_metrics = evaluate.evaluate(-1)
    
    # Save the evaluation
    eval_df.to_csv(os.path.join(evaluation_path, f'{MODEL_NAME}_eval.csv'), index=False)
    protein_metrics.to_csv(os.path.join(evaluation_path, f'{MODEL_NAME}_protein_metrics.csv'),index=False)
    
    print(eval_df)
    print(protein_metrics)
    
    return eval_df, protein_metrics

if __name__ == '__main__':
    tensor_root_dir = r'./data/Processed_K50_dG_datasets/training_data'
    mutations_root_dir = r'./data/Processed_K50_dG_datasets/mutation_datasets'
    evaluation_path = './Megascale-fineTuning/evaluation/'
    CFG.dropout_rate = DROP_OUT
    evaluat_model()

