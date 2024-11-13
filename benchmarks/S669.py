import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./')
import torch
# dataloaders
import torch.utils.data as data
from Utils.train_utils import  get_graph, get_unfolded_graph, load_checkpoint
from model.hydro_net import PEM
from model.model_cfg import CFG
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class S669_ds(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.coords = torch.load(os.path.join(data_dir, "all_coords.pt"))
        self.masks = torch.load(os.path.join(data_dir, "all_masks.pt"))
        self.proT5_emb = torch.load(os.path.join(data_dir, "all_proT5_mut_embs.pt"))
        self.proT5_wt = torch.load(os.path.join(data_dir, "all_proT5_wild_embs.pt"))
        self.sequences = torch.load(os.path.join(data_dir, "all_sequences.pt"))
        self.one_hot = torch.load(os.path.join(data_dir, "all_one_hot.pt"))
        self.wt_one_hot = torch.load(os.path.join(data_dir, "all_wildtype_one_hot.pt"))
        self.mutation_seq = torch.load(os.path.join(data_dir, "all_mutations.pt"))
        self.ids = torch.load(os.path.join(data_dir, "all_ids.pt"))
        self.muts = torch.load(os.path.join(data_dir, "all_mutations_seq.pt"))
        
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return idx ,self.coords[idx], self.masks[idx], self.proT5_emb[idx], self.mutation_seq[idx],\
            self.one_hot[idx], self.ids[idx], self.muts[idx], self.wt_one_hot[idx], self.proT5_wt[idx]
    

def create_dataloader(data_dir, batch_size):
    dataset = S669_ds(data_dir)
    dataloader = data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

def test_dataloader():
    data_dir = './data/S669/'
    batch_size = 1
    dataloader = create_dataloader(data_dir, batch_size)
    for i, (coords, masks, proT5_embs, mutation_seq) in enumerate(dataloader):
        print(f"Batch {i}")
        print(f"Coords: {coords}")
        print(f"Masks: {masks}")
        print(f"ProT5 Embeddings: {proT5_embs}")
        print(f"Sequences: {mutation_seq}")
        print("\n")
        if i == 2:
            break
        
def inference(model, dataloader, inf_dir) :
    inf_df = pd.DataFrame()
    model.eval()
    with torch.no_grad():
        for i, (idx, coords, masks, proT5_embs, mutation_seq,\
            one_hot, protein_id, mut,wt_one_hot,proT5_wt) in enumerate(tqdm(dataloader)):
             # move to device
            coords = coords.to(device)
            masks = masks.to(device)
            proT5_embs = proT5_embs.to(device)
            one_hot = one_hot.to(device)
            wt_one_hot = wt_one_hot.to(device)
            proT5_wt = proT5_wt.to(device)
            # get wild type graph
            wt_graphs = torch.stack([torch.stack([get_graph(coords[i], wt_one_hot[i], proT5_wt[i], masks[i]), get_unfolded_graph(coords[i], wt_one_hot[i], proT5_wt[i], masks[i])]) for i in range(len(mutation_seq))])                  
            wt_folded_graphs = wt_graphs[:,0]
            wt_unfolded_graphs = wt_graphs[:,1]
            # get graph
            mut_graphs = torch.stack([torch.stack([get_graph(coords[i], one_hot[i], proT5_embs[i], masks[i]), get_unfolded_graph(coords[i], one_hot[i], proT5_embs[i], masks[i])]) for i in range(len(mutation_seq))])
            mut_folded_graphs = mut_graphs[:,0]
            mut_unfolded_graphs = mut_graphs[:,1]
            # forward pass
            mut_folded_energy = model(mut_folded_graphs)
            mut_unfolded_energy = model(mut_unfolded_graphs)
            
            wt_folded_energy = model(wt_folded_graphs)
            wt_unfolded_energy = model(wt_unfolded_graphs)
            
            mut_delta_G = mut_unfolded_energy - mut_folded_energy
            wt_delta_G =  wt_unfolded_energy - wt_folded_energy
            ddG = mut_delta_G - wt_delta_G
            res = {"index":idx[0].item(), "id":protein_id[0], "mut":mutation_seq[0], "mut_folded_energy":mut_folded_energy.item(), "mut_unfolded_energy":mut_unfolded_energy.item(), "mut_delta_G":mut_delta_G.item(), "wt_folded_energy":wt_folded_energy.item(), "wt_unfolded_energy":wt_unfolded_energy.item(), "wt_delta_G":wt_delta_G.item(), "ddG":ddG.item()}
            inf_df = inf_df.append(res, ignore_index=True)
    inf_df.to_csv(inf_dir+"inference_results.csv")
    
    return inf_df
            
def print_stat(inf_df):
    folder_path = './data/S669/'
    df_path = folder_path + 'Data_s669_with_predictions.csv'
    df = pd.read_csv(df_path)

    df_deepEF = inf_df
    df['deepEF'] = df_deepEF['ddG']

    print(f"Pearson correlatoin: {df[['DDG_checked_dir', 'deepEF']].corr(method='pearson').iloc[0,1]}")
    print(f"MAE: {np.abs(df['DDG_checked_dir'] - df['deepEF']).mean()}")
    print(f"RMSE: {np.sqrt(((df['DDG_checked_dir'] - df['deepEF'])**2).mean())}")    
            

def main():
    # test_dataloader()
    data_dir = './data/S669/'
    inf_dir = './data/S669/res/'
    # Load model
    # model_path = './res/trianed_models-no_exdu_nosigmoid/18_final_model.pt'
    model_path ='./Megascale-fineTuning/models/PEM_full_trained-PEM_fine_tuned-trianed_models-cycle_per_norm_SMschedulerkfolds/kf_0_epoch_10.pt'
    # model_path = './Megascale-fineTuning/models/PEM_full_trained/20.pt'
    # model_path = './res/trianed_models-cycle_per/10_final_model.pt'
    # model_path = './res/trianed_models-cycle_per_norm/5_final_model.pt'
    # model_path = './res/trianed_models-cycle_per_norm_SM/3_final_model.pt'
    model_name = model_path.split('/')[-2]
    # create output directory
    inf_dir = inf_dir + model_name + '/'
    if not os.path.exists(inf_dir):
        os.makedirs(inf_dir)
    model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef).to(device) 
    try:
        model_dict = torch.load(model_path,map_location=device)
        model.load_state_dict(model_dict['model_state_dict'])
    except:
        model.load_state_dict(torch.load(model_path)) 
    # Load data
    dataloader = create_dataloader(data_dir, 1)
    # Inference
    inf_df = inference(model, dataloader, inf_dir)
    # Print statistics  
    print_stat(inf_df)
    
if __name__ == "__main__":
    main()