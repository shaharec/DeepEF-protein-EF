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
MODEL_NAME = 'PEM_fine_tuned-'+BASE_MODEL_NAME if FREEZE_LAYERS else 'PEM_full_trained-'+BASE_MODEL_NAME
MODEL_NAME += 'kf'
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


# Trainer class

class Trainer():
    def __init__(self, model, train_ds, val_ds, device = DEVICE):
        self.model = model.to(device)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.device = device
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=5, verbose=True)
        self.model.to(self.device)
        self.mini_batch_size = MINI_BATCH_SIZE
        self.model_name = 'PEM_fine_tuned' if FREEZE_LAYERS else 'PEM_full_trained'

    def train(self, epochs = 10, kf = 0):
        """
        Train the model
        args:
        epochs: int, number of epochs
        kf: int, kfold number
        """
        if not DEBUG:
            run = wandb.init(project='KF-1MUT-MegaScaleFineTuning', config=config, name=MODEL_NAME + f'Kfold{kf}')
        
        # Freeze the layers and only train the last layer
        if FREEZE_LAYERS:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc2.parameters():
                param.requires_grad = True
            for param in self.model.fc1.parameters():
                param.requires_grad = True
        running_loss = 0
        wandb_step = 0
        for epoch in range(epochs):
            self.model.train()
            for i, batch in enumerate(tqdm(self.train_ds, desc=f'Training Epoch: {epoch}')):
                batch = normalize_batch(batch, True)
                batch_loss = 0
                batch_idx = 1
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    self.optimizer.zero_grad()
                    output,u_energy,f_energy = self.get_deltaG(batch, j)
                    delta_g = batch['delta_g'][0, j: j + self.mini_batch_size].to(self.device)
                    l1_loss = self.criterion(output, delta_g)
                    if FREEZE_LAYERS:
                        reg_loss = REG_LAMBDA * (F.mse_loss(self.model.fc1.weight,torch.zeros_like(self.model.fc1.weight)) + F.mse_loss(self.model.fc2.weight,torch.zeros_like(self.model.fc2.weight)))
                    else:
                        reg_loss = REG_LAMBDA * sum([F.mse_loss(param,torch.zeros_like(param)) for param in self.model.parameters()])
                    energys = torch.cat((u_energy,f_energy),dim=0)
                    energy_reg = REG_LAMBDA * (F.mse_loss(energys,torch.zeros_like(energys)))
                    loss = l1_loss + reg_loss +energy_reg
                    loss.backward()
                    self.optimizer.step()
                    batch_loss += loss.item()
                    wandb_step += 1
                    wandb_log({'loss': loss.item(), 'epoch': epoch, 'batch': i,'l1_loss': l1_loss.item(), 'reg_loss': reg_loss.item(), 'energy_reg': energy_reg.item(), 'wandb_step': wandb_step},run)
                batch_loss /= batch_idx
                running_loss += batch_loss
                if (i+1) % 100 == 0:
                    wandb_log({'epoch': epoch, 'running_loss': running_loss/100},run)
                    running_loss = 0
                    
            # save the model
            if not DEBUG:
                torch.save(self.model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME, f'kf_{kf}_epoch_{epoch}.pt'))
            
            pc_corr = self.validate(epoch,run)
            # update the learning rate
            self.scheduler.step(pc_corr)
            current_lr = self.optimizer.param_groups[0]['lr']
            wandb_log({'epoch': epoch, 'lr': current_lr}, run)
        return self.model, pc_corr

    def validate(self, epoch, run = None):
        """
        Validate the model
        args:
        epoch: int, the current epoch
        run: wandb run object
        returns:
        pc_corr: float, pearson correlation
        """
        self.model.eval()
        val_loss = 0
        val_dg = torch.tensor([],device=self.device)
        val_dg_pred = torch.tensor([],device=self.device)
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_ds,desc=f'Validation Epoch: {epoch}')):
                batch = normalize_batch(batch, True)
                batch_loss = 0
                batch_idx = 1
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    batch_idx += 1
                    output,u_energy,f_energy = self.get_deltaG(batch, j)
                    delta_g = batch['delta_g'][0, j: j + self.mini_batch_size].to(self.device)
                    loss = self.criterion(output,delta_g)
                    energys = torch.cat((u_energy,f_energy),dim=0)
                    energy_reg = REG_LAMBDA * (F.mse_loss(energys,torch.zeros_like(energys)))
                    loss += energy_reg
                    batch_loss += loss.item()
                    val_dg = torch.cat((val_dg, delta_g), dim=0)
                    val_dg_pred = torch.cat((val_dg_pred, output), dim=0)
                batch_loss /= batch_idx
            val_loss += batch_loss
        val_loss /= len(self.val_ds)
        print(f'Validation Loss: {val_loss}')
        pc_corr = torch.corrcoef(torch.cat((val_dg[None,:],val_dg_pred[None,:])))[0, 1]
        wandb_log({'val_loss': val_loss,'epoch': epoch, 'pc_corr': pc_corr},run)
        self.model.train()
        return pc_corr
        
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


def run_training():
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)

    results = {}

    prot_ds = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                          mutations_root_dir=mutations_root_dir, train=True)
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(prot_ds)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = Subset(prot_ds, train_ids)
        test_subsampler = Subset(prot_ds, test_ids)
    
        # Create the dataloaders
        train_ds = DataLoader(train_subsampler, batch_size=1, shuffle=False)
        val_ds = DataLoader(test_subsampler, batch_size=1, shuffle=False)
    
        # Create the model
        model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,dropout_rate = CFG.dropout_rate).to(DEVICE)
        if PRETRAINED: 
            try:
                model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)
            except:
                model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
        
        # Train the model
        trainer = Trainer(model, train_ds, val_ds)
        model, pc_corr = trainer.train(epochs = EPOCHS, kf=fold)
        results[fold] = pc_corr
        wandb.finish()
        
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        
def get_valid_proteins(val_ds):
    # create dataframe and append the name of the protein and the mutations
    df = pd.DataFrame(columns=['name', 'mutations'])
    for i, batch in enumerate(val_ds):
        df = df.append({'name': batch['name'][0]}, ignore_index=True)
    
    df.to_csv('validation_proteins_mutations.csv', index=False)
    
    return df
if __name__ == '__main__':
    tensor_root_dir = r'./data/Processed_K50_dG_datasets/training_data'
    mutations_root_dir = r'./data/Processed_K50_dG_datasets/mutation_datasets'
    CFG.dropout_rate = DROP_OUT
    run_training()
