from model.data_loader import fetch_dataloader,fetch_inference_loader
from model.data_loader import params as data_params
from model.model_cfg import CFG
from model.hydro_net import PEM
from model.net import params as model_params
from train_utils import *
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from tqdm import tqdm
import gc
import time
import sys
import pandas as pd
import wandb

# Set the default data type to float32
torch.set_default_dtype(CFG.torch_default_dtype)
# torch.autograd.set_detect_anomaly(True)
# CFG.debug = True
# CFG.clip_grad_norm = True
# Set wandb
if not CFG.debug:
    wandb.init(project="Evaluate base training",name = 'cycel 2-5')
if CFG.debug:
   CFG.model_path = "./res/debug/"
   CFG.results_path = './res/debug/results-debug/'
   print('**** Debug mode ****')



def get_noised_proteins(data,device):
    """
    Returns a noised version of the protein data.
    """
    id, crd_backbone, mask, seq_one_hot, seq,ang_backbone, ang,\
                proT5_emb, proT5_mut,seq_mut, crd_decoy, mask_crd_decoy, seq_crd_decoy,proT5_cycle1,\
                proT5_cycle2, proT5_cycle3, proT5_cycle4,proT5_cycle5,proT5_cycle6 = data
            
    # wild type and mutant type
    Xjf = crd_backbone.to(device) # wilde type structure folded
    Xju = torch.clone(Xjf).to(device) # wilde type structure unfolded
    Xcd = torch.clone(crd_decoy).to(device) # decoy structure
    Xcy1 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    Xcy2 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    Xcy3 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    Xcy4 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    
    # change the decoy len to match the native len
    if Xcd.shape[1] > Xjf.shape[1]:
        Xcd = Xcd[:,:Xjf.shape[1],:,:]
        mask_crd_decoy = mask_crd_decoy[:,:Xjf.shape[1]]
    elif Xcd.shape[1] < Xjf.shape[1]:
        # add zeros to the end of the decoy
        Xcd = torch.cat((Xcd, torch.zeros(Xjf.shape[0],Xjf.shape[1] - Xcd.shape[1], *Xcd.shape[2:]).to(device)), dim=1)
        mask_crd_decoy = torch.cat((mask_crd_decoy, torch.zeros(mask.shape[0],mask.shape[1] - mask_crd_decoy.shape[1])), dim=1)
        
    
    # native structure and decoy structure
    Xd = torch.clone(Xjf).to(device)
    Xdu = torch.clone(Xjf).to(device)
    seq_one_hot = seq_one_hot.to(device) # [batch_size,20,seq_len]
    
    # create decoy sequence
    seq_decoy,mask_decoy, proT5_emb_decoy = mix_A_acid(seq_one_hot = seq_one_hot, emb=proT5_emb, mask = mask,val_type='train',device=device)
    
    if seq_decoy.shape[1] >CFG.seq_len : # if the sequence is too long, skip it(GPU limitation)
        return None,None,None,None,None,None,None,None,None
    #emb = torch.cat((esm_embed,seq),dim=2)
    emb = seq_one_hot.to(device)
    emb_decoy = seq_decoy.to(device)
    # get the cycle permutation
    cycle_emb1 = get_one_hot(seq[0][-1] + seq[0][:-1]).to(device)
    cycle_emb2 = get_one_hot(seq[0][1:] + seq[0][0]).to(device)
    cycle_emb3 = get_one_hot(seq[0][-2:] + seq[0][:-2]).to(device)
    cycle_emb4 = get_one_hot(seq[0][-5:] + seq[0][:-5]).to(device)
    
    # move proT5_emb to device
    proT5_emb_decoy, proT5_emb, proT5_cycle1, proT5_cycle2, proT5_cycle3, proT5_cycle4 = proT5_emb_decoy.to(device), proT5_emb.to(device), proT5_cycle1.to(device), proT5_cycle2.to(device), proT5_cycle3.to(device), proT5_cycle4.to(device)
    
    # squeeze the data
    Xd, Xjf, Xju, Xcd, Xdu, Xcy1, Xcy2, Xcy3, Xcy4 = Xd.squeeze(), Xjf.squeeze(), Xju.squeeze(), Xcd.squeeze(), Xdu.squeeze(), Xcy1.squeeze(), Xcy2.squeeze(), Xcy3.squeeze(), Xcy4.squeeze()
    emb_decoy, emb = emb_decoy.squeeze(), emb.squeeze()
    mask_decoy, mask, mask_crd_decoy= mask_decoy.squeeze(), mask.squeeze(), mask_crd_decoy.squeeze()
    proT5_emb_decoy, proT5_emb = proT5_emb_decoy.squeeze(), proT5_emb.squeeze()
    proT5_cycle1, proT5_cycle2, proT5_cycle3, proT5_cycle4 = proT5_cycle1.squeeze(), proT5_cycle2.squeeze(), proT5_cycle3.squeeze(), proT5_cycle4.squeeze()
    # get folded graph  
    Xjf = get_graph(Xjf, emb, proT5_emb, mask)
    # get unfolded graph
    Xju = get_unfolded_graph(Xju, emb, proT5_emb, mask)
    # get decoy graph
    Xd, Xcd, Xdu = get_graph(Xd, emb_decoy, proT5_emb_decoy, mask_decoy), get_graph(Xcd, emb, proT5_emb, mask_crd_decoy), get_unfolded_graph(Xdu, emb_decoy, proT5_emb_decoy, mask_decoy)
    # Add cycle permutation
    Xcy1 = get_graph(Xcy1, cycle_emb1, proT5_cycle1, mask)
    Xcy2 = get_graph(Xcy2, cycle_emb2, proT5_cycle2, mask)
    Xcy3 = get_graph(Xcy3, cycle_emb3, proT5_cycle3, mask)
    Xcy4 = get_graph(Xcy4, cycle_emb4, proT5_cycle4, mask)
    # Xjf.requires_grad = True
    # create a batch of Xjf,Xkf,Xju,Xku,x_decoy
    Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2,Xcy3,Xcy4 = Xjf.unsqueeze(0),Xju.unsqueeze(0),Xd.unsqueeze(0), Xcd.unsqueeze(0), Xdu.unsqueeze(0),Xcy1.unsqueeze(0),Xcy2.unsqueeze(0),Xcy3.unsqueeze(0),Xcy4.unsqueeze(0)

    return Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2,Xcy3,Xcy4
    
# define validation function
def validation(model, dataloader, device, epoch, N, optimizer):
    """
    Validation function for the model.
    """
    model.eval()
    results = []
    total_energies = {
        'Ejf': 0, 'Eju': 0, 'Exd': 0, 'Ecd': 0, 'Exdu': 0,
        'Ecy1': 0, 'Ecy2': 0, 'Ecy3': 0, 'Ecy4': 0, 'Ecy5': 0, 'Ecy6': 0
    }
    num_proteins = 0
    
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Evaluation: Epoch {epoch}")
            for index, data in enumerate(tepoch):
                if device.type == "cuda" or device.type == "mps":
                    torch.cuda.empty_cache()
                gc.collect()
                
                Xjf, Xju, Xd, Xcd, Xdu, Xcy1, Xcy2, Xcy3, Xcy4 = get_noised_proteins(data, device)
                if Xjf is None:
                    continue
                X = torch.cat((Xjf, Xju, Xd, Xcd, Xdu, Xcy1, Xcy2, Xcy3, Xcy4), dim=0)
                
                with torch.amp.autocast(device_type="cuda", dtype=CFG.precision):
                    E = model(X)
                    Ejf, Eju, Exd, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4 = E[0], E[1], E[2], E[3], E[4], E[5], E[6], E[7], E[8]
                
                loss, lossd, lossg, lossc = criterion(Ejf, Eju, Exd, Xju, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4, with_grad = False)
                
                # Get protein ID
                protein_id = data[0][0]  # Assuming the first element of data contains the protein ID
                
                # Store results
                result_dict = {
                    'protein_id': protein_id,
                    'Ejf': Ejf.item(),
                    'Eju': Eju.item(),
                    'Exd': Exd.item(),
                    'Ecd': Ecd.item(),
                    'Exdu': Exdu.item(),
                    'Ecy1': Ecy1.item(),
                    'Ecy2': Ecy2.item(),
                    'Ecy3': Ecy3.item(),
                    'Ecy4': Ecy4.item(),
                    'loss': loss.item(),
                    'lossd': lossd.item(),
                    'lossg': lossg.item(),
                    'lossc': lossc.item()
                }
                results.append(result_dict)
                
                num_proteins += 1
                
                # Log individual protein results to wandb
                if not CFG.debug:
                    wandb.log(result_dict)
                
                tepoch.set_postfix({"protein": protein_id})
    
    # Calculate and log average energies
    avg_energies = {f"avg_{k}": v / num_proteins for k, v in total_energies.items()}
    
    if not CFG.debug:
        wandb.log(avg_energies)
    
    return results

def save_results_to_csv(results, filename):
    """
    Save the evaluation results to a CSV file.
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def evaluate_and_save(model, dataloader, device, epoch, N, optimizer,dataset):
    """
    Evaluate the model and save the results.
    """
    results = validation(model, dataloader, device, epoch, N, optimizer)
    csv_filename = f"{CFG.results_path}evaluation_results_dataset_{dataset}.csv"
    save_results_to_csv(results, csv_filename)



def gradient_penalty(X_native, E_native):
    """Implementing the lossg equation:
        The gradient of a wild type structure should be close to zero.
        Therefore we will add it to the loss as lossg"""
    partial_dx_native = torch.autograd.grad(outputs=E_native, inputs=X_native,
                                            grad_outputs=torch.ones_like(E_native),
                                            create_graph=True, retain_graph=True)[0]
    # Use mse loss
    lossg = torch.mean(partial_dx_native**2)
    return lossg

def criterion(Ejf, Eju, Exd, X_native, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4, with_grad = True , reg_alpha = CFG.reg_alpha):
    """
    The loss function for the model corresponds to 3 main losses:
    1. lossg: the partial derivative of the energy with respect to the native structure
    2. lossd: the energy of the native structure divided by the decoy energy
    3. lossc: the energy softplus function for the native and mutant structure(unfolded and folded)
    Args:
        Ejf (tensor): The energy of the folded native structure
        Ekf (tensor): The energy of the folded mutant structure
        Eju (tensor): The energy of the unfolded native structure
        Eku (tensor): The energy of the unfolded mutant structure
        Exd (tensor): The energy of the decoy sequence
        Ecd (tensor): The energy of the decoy structure
        Exdu (tensor): The energy of the decoy structure unfolded
        Ecy1 (tensor): The energy of the cycle permutation structure first amino acid
        Ecy2 (tensor): The energy of the cycle permutation structure last amino acid
        Ecy3 (tensor): The energy of the cycle permutation structure last two amino acid
        Ecy4 (tensor): The energy of the cycle permutation structure last five amino acid
    output:
        loss (tensor): The loss of the model
        lossd (tensor): The loss of the model due to the energy of the native structure divided by the decoy energy
        lossg (tensor): The loss of the model due to the partial derivative of the energy with respect to the native structure
        lossc (tensor): The loss of the model due to the energy softplus function for the native and mutant structure(unfolded and folded)
    """
    lossg = gradient_penalty(X_native, Ejf) if with_grad else torch.tensor(0.0).to(Ejf.device)
    lossd = lossd_fucntion(Ejf, Exd, Ecd, Exdu, Eju, Ecy1, Ecy2, Ecy3, Ecy4)
    # lossc will be regularization term of sum of squered energys 
    lossc = (torch.cat([Ejf.unsqueeze(0)[None,:], Eju.unsqueeze(0)[None,:],
                        Exd.unsqueeze(0)[None,:], Ecd.unsqueeze(0)[None,:],
                        Exdu.unsqueeze(0)[None,:], Ecy1.unsqueeze(0)[None,:],
                        Ecy2.unsqueeze(0)[None,:], Ecy3.unsqueeze(0)[None,:],
                        Ecy4.unsqueeze(0)[None,:]])**2).mean()
    lossc = reg_alpha * lossc
    
    return lossd+lossg+lossc , lossd, lossg, lossc
  
def lossd_fucntion(Ejf, Exd, Ecd, Exdu, Eju, Ecy1, Ecy2, Ecy3, Ecy4):
    """Decoy loss:
    - the energy of a decoy sequece is greater than the energy of the wild-type structure (Ejf<Exd)
    - the energy of a decoy structure is greater than the energy of the wild-type structure (Ejf<Ecd)
    - the energy of a folded decoy is greater than the energy of an unfolded decoy (Exdu<Exd)
    - the energy of a decoy structure is greater than the energy of the unfolded native structure (Eju<Ecd)
    - the energy of the wild-type structure is lower than the energy of the cycle permutation (Ejf<Ecy1)
    - the energy of the wild-type structure is lower than the energy of the cycle permutation (Ejf<Ecy2)
    - the energy of the wild-type structure is lower than the energy of the cycle permutation (Ejf<Ecy3)
    - the energy of the wild-type structure is lower than the energy of the cycle permutation (Ejf<Ecy4)
    """
    # loss_decoy = lambda x,y: torch.log((x+1) / (y+1) +1)
    loss_decoy = lambda x,y: x - y
    loss = torch.cat([loss_decoy(Ejf, Exd).unsqueeze(0)[None,:], loss_decoy(Ejf, Ecd).unsqueeze(0)[None,:], 
                      loss_decoy(Eju, Ecd).unsqueeze(0)[None,:], loss_decoy(Ejf, Eju).unsqueeze(0)[None,:], 
                      loss_decoy(Ejf, Ecy1).unsqueeze(0)[None,:], loss_decoy(Ejf, Ecy2).unsqueeze(0)[None,:],
                      loss_decoy(Ejf, Ecy3).unsqueeze(0)[None,:], loss_decoy(Ejf, Ecy4).unsqueeze(0)[None,:]])
    loss = torch.mean(loss)
    return loss

    
def main():
    print('***Start main function***')
    print('***load the data with dataloader***')
    d_params = data_params(num_workers =CFG.num_workers, batch_size=CFG.batch_size,cuda=CFG.cuda,constraint=CFG.constraint, 
                           debug=CFG.debug,dataset='scn',LLM_EMB=True)
    train_loader, valid_loader,test_loader = fetch_dataloader(data_dir=CFG.data_path, params=d_params)
    # Build the model
    print('***Build the model***')
    model = PEM(layers=CFG.num_layers,gaussian_coef=CFG.gaussian_coef,dropout_rate=CFG.dropout_rate).to(CFG.device)
    model.name = "PEM-With LLM embedding"
    model.energy_epsilon = 1e-6
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
    # Define the learning rate scheduler based on loss
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    # configurate wandb
    wandb_config(wandb, model, optimizer, scheduler, train_loader,
                CFG.model_path,CFG.reg_alpha,CFG.gaussian_coef,CFG.lr,
                CFG.num_layers,CFG.dropout_rate,CFG.precision)
    # Run training
    print('***Start training***')
    # Load the best model
    load_checkpoint(CFG.model_path + "best_model.pt", model, optimizer, CFG.device)
    
    # Evaluate and save resulys for training set
    print("Evaluating training set...")
    evaluate_and_save(model, train_loader, CFG.device, -1, CFG.N, optimizer,'training'
                      )
    # Evaluate and save results for validation set
    print("Evaluating validation set...")
    evaluate_and_save(model, valid_loader, CFG.device, -1, CFG.N, optimizer,'validation')
    
    # Evaluate and save results for test set
    print("Evaluating test set...")
    evaluate_and_save(model, test_loader, CFG.device, -1, CFG.N, optimizer,'test')
    
    return 1

    
def print_par(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)
   
if __name__ == '__main__':
    if not CFG.debug:
        CFG.model_path = './res/trianed_models-cycle2_5/'
        CFG.results_path = './res/trianed_models-cycle2_5/'
    # CFG.data_path = './data/casp12_data_30/'
    CFG.dropout_rate = 0.3
    CFG.gaussian_coef = -0.08
    CFG.reg_alpha = 0.1
    main()
