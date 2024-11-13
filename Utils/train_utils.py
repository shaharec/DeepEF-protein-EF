import torch
import torch.nn.functional as F
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
from model.model_cfg import CFG
import constants as C

# amino acid one hot map
AA_MAP = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19
}

def save_checkpoint(epoch, model, optimizer,loss,val_loss,path):
    """
    Save the model check point
    inputs:
        epoch (int): number of epoch
        model(torch.model): model
        optimizer(torch.optim): torch optimizer
        loss(tensor) : loss function value
        val_loss(tensor) : validation loss function value
        path (str) : path to save the model
    """
    os.makedirs(str(Path(path).parent), exist_ok=True)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'valid_loss': val_loss,
            }, path)
   
def load_checkpoint(path,model,optimizer=None,device=CFG.device):
    """
    Load the model check point
    inputs:
        path (str) : path to load the model
        model(torch.model): model
        optimizer(torch.optim): torch optimizer
        device (str) : device to load the model
    """ 
    
    model_dict = torch.load(path,map_location=device)
    print(f"Loaded model from {path}")
    # print(f"Epoch: {dict['epoch']},loss: {dict['loss']},valid_loss: {dict['valid_loss']}")
    model.load_state_dict(model_dict['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(model_dict['optimizer_state_dict'])
    return model,optimizer,model_dict['epoch'],model_dict['loss'],model_dict['valid_loss']
    
def validation_plots(Exd,Exn,seq_len,type,epoch):
    """
    Plot the validation data
    inputs:
        Exd (tensor) : validation data
        Exn (tensor) : validation data
        seq_len (int) : sequence length
        type (str) : type of the plot
    """
    # create the directory if not exist
    os.makedirs(CFG.results_path+'plots', exist_ok=True)
    plot_dir = CFG.results_path+'plots'
    # Plot the validation data
    fig, ax = plt.subplots()
    ax.set_title(f'Validation data for {type},number of sequences: {len(Exd)}')
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Energy')
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    ax.plot(seq_len, Exd, marker='o', linestyle='', ms=3, label='decoy')
    ax.plot(seq_len, Exn, marker='o', linestyle='', ms=3, label='native')
    ax.legend()

    #plt.show()
    plt.savefig(f'{plot_dir}/E_len_{type}.png')
    
    plt.close()
    
    # plot validation energy delta
    delta_E = np.array([Exd[i]-Exn[i] for i in range(len(seq_len))])
    seq_len = np.array(seq_len)
    fig, ax = plt.subplots()
    ax.set_title(f'Validation data for {type},number of sequences: {len(Exd)}')
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Energy delta log')
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    ax.plot(seq_len[delta_E>=0], np.log(delta_E[delta_E>=0]+1), marker='o', linestyle='', ms=3, label='positive')
    ax.plot(seq_len[delta_E<0], -1*np.log(-1*delta_E[delta_E<0] +1), marker='o', linestyle='', ms=3, label='negative')
    ax.legend()

    #plt.show()
    plt.savefig(f'{plot_dir}/epoch-{epoch}-Edelta_len_{type}.png')
    
    plt.close()

def mix_A_acid(seq_one_hot,emb,mask,val_type,device):
    """ mix the amino acid sequence"""
    if val_type == 'robust' or val_type == 'train':
        mix_index = torch.randperm(seq_one_hot.shape[1])
        seq_decoy = torch.clone(seq_one_hot[:,mix_index,:]).to(device)
        mask_decoy = torch.clone(mask[:,mix_index]).to(device)
        emb_decoy = torch.clone(emb[:,mix_index,:]).to(device)
    else: 
        mix_index = torch.randperm(seq_one_hot.shape[1])[:2]
        mask_decoy = torch.clone(mask).to(device)
        seq_decoy = torch.clone(seq_one_hot).to(device)
        emb_decoy = torch.clone(emb).to(device)
        seq_decoy[:,mix_index[0],:], seq_decoy[:,[1],:] = seq_decoy[:,mix_index[1],:], seq_decoy[:,[0],:]
        mask_decoy[:,mix_index[0]] ,mask_decoy[:,[1]] = mask_decoy[:,mix_index[1]], mask_decoy[:,[0]]
        emb_decoy[:,mix_index[0],:], emb_decoy[:,[1],:] = emb_decoy[:,mix_index[1],:], emb_decoy[:,[0],:]
    return seq_decoy, mask_decoy, emb_decoy
        
def pad_image(image,desired_size = (500,23)):
    """ pad the image to desired size"""
    # Pad the image to 500x500
    desired_height, desired_width = desired_size
    pad_height = desired_height - image.size(0)
    pad_width = desired_width - image.size(1)

    # Compute the amount of padding on each side
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    # Apply padding using torch.nn.functional.pad
    padded_image = F.pad(image, (left_pad, right_pad, top_pad, bottom_pad))
    
    return padded_image

def interpolate_image(Xn):
    """ interpolate the image"""
    # Interpolate the image to 500x500
    Xn = F.interpolate(Xn.unsqueeze(0), size=500)
    Xn = Xn.squeeze(0)
    return Xn

def diff_data(model, optimizer, dataloader, device,epoch,N,valid_loader):
    all_Xn = torch.tensor([])
    all_Xn_int = torch.tensor([])
    n = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        for index, data in enumerate(tepoch):
            # set progress bar description
            tepoch.set_description(f"Epoch {epoch}")
            # Clean the GPU cache
            torch.cuda.empty_cache()
            gc.collect()
            # get the inputs; data is a list of [inputs, labels]   
            id, Xn, mask, seq_one_hot, seq,ang_backbone, ang, dist_matrix = data
            
            mask_flag = torch.where(mask==1,0,1).sum()==0
            if((seq_one_hot.shape[1] <= 500) & mask_flag):
                Xn = Xn.squeeze()
                seq_one_hot = seq_one_hot.squeeze() 
                Xn = Xn[:,1,:] # take only the first 128 residues
                mean, std, var = torch.mean(Xn), torch.std(Xn), torch.var(Xn) 
                Xn  = (Xn-mean)/std
                Xn = torch.cat((Xn,seq_one_hot),dim=1)
                Xn_pad = pad_image(Xn)
                Xn_interpolat = interpolate_image(Xn.swapaxes(0,1)).swapaxes(0,1)
                all_Xn_int = torch.cat((all_Xn_int,Xn_interpolat.unsqueeze(dim=0)),dim=0)
                all_Xn = torch.cat((all_Xn,Xn_pad.unsqueeze(dim=0)),dim=0)
                n = n+1
                # torch.save(all_Xn,"./all_Xn.pt")
            tepoch.set_postfix({"n":n})
        
        torch.save(all_Xn_int,"./all_Xn_int.pt")
        torch.save(all_Xn,"./all_Xn_padded.pt")

def get_graph(x, one_hot, emb, mask, gaussian_coef=CFG.gaussian_coef):
    """Get graph representation of protein"""
    D = get_dist_matrix(x) # N,N,16
    D = torch.relu(torch.exp(gaussian_coef*D**2))
    # remove masks values
    mask_index = torch.where(mask == 0)
    D[mask_index[0],:,:] = 0
    D[:,mask_index[0],:] = 0
    # get bonded features
    Fb = get_bonded_features(D) # N,32
    # sum over the atoms,
    D = D.sum(dim=1) #N,16
    D = F.normalize(D,p=2,dim=0)
    emb = F.normalize(emb,p=2,dim=0)
    Fh = torch.cat([D,Fb,emb,one_hot],dim=1) #N,16+32+emb_size
    
    return Fh

def get_unfolded_graph(x, one_hot, emb, mask, gaussian_coef=CFG.gaussian_coef):
    """Get graph representation of a unfolded protein"""
    D = get_dist_matrix(x) # N,N,16
    D = torch.relu(torch.exp(gaussian_coef*D**2))
    # remove masks values
    mask_index = torch.where(mask == 0)
    D[mask_index[0],:,:] = 0
    D[:,mask_index[0],:] = 0
    # remove all values except the diagonal
    D = zero_except_udiagonal(D)
    # get bonded features
    Fb = get_bonded_features(D) # N,32
    # sum over the atoms
    D = D.sum(dim=1) #N,16
    D = F.normalize(D,p=2,dim=0)
    emb = F.normalize(emb,p=2,dim=0)
    Fh = torch.cat([D,Fb,emb,one_hot],dim=1) #N,16+32+emb_size
    
    return Fh

def get_bonded_features(D):
    """Get bonded features from distance matrix"""
    n_range = torch.arange(D.shape[0])
    # get above and below diagonal
    f1, f2 = D[n_range[:-1], n_range[1:]], D[n_range[1:], n_range[:-1]]
    # pad with zeros
    zero_row = torch.zeros((1, D.shape[-1])).to(D.device)
    f1 = torch.cat([f1, zero_row], dim=0)
    f2 = torch.cat([zero_row, f2], dim=0)
    # concat features
    Fb = torch.cat([f1, f2], dim=-1)
    return Fb # N,32   
    
      
def get_dist_matrix(Xd):
    """
    Return the node distence matrix
    Args:
        Xd (tensor):X embeded [n_nodes ,num_atoms=4,new_cords_size]
    Returns:
        tensor : [n_nodes,n_nodes ,atom_dist=16] tensor
    """
    N_residu, N_atoms, coords_size = Xd.shape
    Xd = Xd.reshape(N_residu*N_atoms, coords_size)
    D = torch.cdist(Xd, Xd, p=2)
    D = D.reshape(N_residu, N_atoms, N_residu, N_atoms)
    D = torch.swapaxes(D,1,2)
    D = D.reshape(N_residu, N_residu, N_atoms*N_atoms)
    return D

def add_gaussian_noise(X_native,sigma):
    """ add gaussian noise to the native structure"""
    device = X_native.device
    noise = (torch.randn(X_native.shape)*sigma).clone().to(device)
    X_decoy = X_native + noise
    return X_decoy

def wandb_config(wandb, model, optimizer, scheduler, dataloader,
                 model_path = CFG.model_path,reg_alpha = CFG.reg_alpha,gaussian_coef = CFG.gaussian_coef,
                 lr = CFG.lr,num_layers = CFG.num_layers,dropout_rate = CFG.dropout_rate,precision = CFG.precision):
    """wandb_config """
    if not CFG.debug:
        wandb.config.learning_rate = optimizer.param_groups[0]['lr']
        wandb.config.batch_size = CFG.batch_size
        wandb.config.epochs = CFG.num_epochs
        wandb.config.optimizer = type(optimizer).__name__
        wandb.config.scheduler = type(scheduler).__name__
        wandb.config.model = type(model).__name__
        wandb.config.dataset = type(dataloader.dataset).__name__
        wandb.config.wd = CFG.wd
        wandb.config.model_path = model_path
        wandb.config.reg_alpha = reg_alpha
        wandb.config.gaussian_coef = gaussian_coef
        wandb.config.lr = lr
        wandb.config.num_layers = num_layers
        wandb.config.dropout_rate = dropout_rate
        wandb.config.precision = precision
        

def zero_except_udiagonal(D):
    """Zero all values except the diagonal and its neighbors"""
    n_range = torch.arange(D.shape[0])
    f1, f2 = D[n_range[:-1], n_range[1:]], D[n_range[1:], n_range[:-1]]
    diag = D[n_range, n_range]
    D[:, :, :] = 0
    D[n_range[:-1], n_range[1:]] = f1
    D[n_range[1:], n_range[:-1]] = f2
    D[n_range, n_range] = diag
    return D

def get_one_hot(seq):
  """get one hot from sequence"""
  seq_one_hot = torch.zeros((len(seq),20))
  for i,a in enumerate(seq):
    if a in AA_MAP:
        seq_one_hot[i][AA_MAP[a]] = 1
  return seq_one_hot

def add_cb(crd_coords):
        """
        Add the Cbeta atom to the coordinates
        Args:
            crd_coords (tensor): tensor of shape [n_residues,3,3]

        Returns:
            crd_coords: tensor shape [n_residues,4,3]
        """
        # Get the coordinates of the backbone atoms
        N, CA, C = crd_coords[:, 0], crd_coords[:, 1], crd_coords[:, 2]
        # CB = CA + c1*(N-CA) + c2*(C-CA) + c3* (N-CA)x(C-CA)
        CAmN = N - CA
        # CAmN = CAmN / torch.sqrt(CAmN ** 2).sum(dim=2, keepdim=True)
        CAmC = C - CA
        # CAmC = CAmC / torch.sqrt(CAmC ** 2).sum(dim=2, keepdim=True)
        ANxAC = torch.cross(CAmN, CAmC, dim=1)

        A = torch.cat((CAmN.reshape(-1, 1), CAmC.reshape(-1, 1), ANxAC.reshape(-1, 1)), dim=1)
        c = torch.tensor([0.5507, 0.5354, -0.5691]) / 100  # torch.tensor([1.1930, 1.2106, -2.7906]) #
        b = (A @ c).reshape(-1,3)
        CB = CA - b
      
        # Add Cbeta coordinates to existing coordinates array
        crd_coords = torch.cat((crd_coords, CB.unsqueeze(1)), dim=1)
        return crd_coords
    
def Add_random_step(X, h = CFG.h):
    """Add random step to the coordinates
    Args:
        X (tensor): tensor of coordinats of shape [n_residues,4,3]
        corrds_index (int): number of coordinates to add
        h (float): step size
    """
    # Add random step to the coordinates
    v = torch.randn(X.shape).to(X.device)
    X1 = X + h * v
    X2 = X - h * v
    return X1, X2

def print_max_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            max_grad = param.grad.data.abs().max().item()
            print(f"Max gradient for {name}: {max_grad}")
            
            
def get_noised_proteins(data,device, config = CFG):
    """
    Returns a noised version of the protein data.
    """
    id, crd_backbone, mask, seq_one_hot, seq,ang_backbone, ang,\
                proT5_emb, proT5_mut,seq_mut, crd_decoy, mask_crd_decoy, seq_crd_decoy,proT5_cycle1, proT5_cycle2 = data
            
    # wild type and mutant type
    Xjf = crd_backbone.to(device) # wilde type structure folded
    Xju = torch.clone(Xjf).to(device) # wilde type structure unfolded
    Xcd = torch.clone(crd_decoy).to(device) # decoy structure
    Xcy1 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    Xcy2 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    
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
    
    if seq_decoy.shape[1] >config.seq_len : # if the sequence is too long, skip it(GPU limitation)
        return None,None,None,None,None,None,None
    #emb = torch.cat((esm_embed,seq),dim=2)
    emb = seq_one_hot.to(device)
    emb_decoy = seq_decoy.to(device)
    # get the cycle permutation
    cycle_emb1 = get_one_hot(seq[0][-1] + seq[0][:-1]).to(device)
    cycle_emb2 = get_one_hot(seq[0][1:] + seq[0][0]).to(device)
    
    # move proT5_emb to device
    proT5_emb_decoy, proT5_emb, proT5_cycle1, proT5_cycle2 = proT5_emb_decoy.to(device), proT5_emb.to(device), proT5_cycle1.to(device), proT5_cycle2.to(device)
    
    # squeeze the data
    Xd, Xjf, Xju, Xcd, Xdu, Xcy1, Xcy2 = Xd.squeeze(), Xjf.squeeze(), Xju.squeeze(), Xcd.squeeze(), Xdu.squeeze(), Xcy1.squeeze(), Xcy2.squeeze()
    emb_decoy, emb = emb_decoy.squeeze(), emb.squeeze()
    mask_decoy, mask, mask_crd_decoy= mask_decoy.squeeze(), mask.squeeze(), mask_crd_decoy.squeeze()
    proT5_emb_decoy, proT5_emb, proT5_cycle1, proT5_cycle2 = proT5_emb_decoy.squeeze(), proT5_emb.squeeze(), proT5_cycle1.squeeze(), proT5_cycle2.squeeze()
    
    # get folded graph  
    Xjf = get_graph(Xjf, emb, proT5_emb, mask, gaussian_coef=config.gaussian_coef)
    # get unfolded graph
    Xju = get_unfolded_graph(Xju, emb, proT5_emb, mask, gaussian_coef=config.gaussian_coef)
    # get decoy graph
    Xd, Xcd, Xdu = get_graph(Xd, emb_decoy, proT5_emb_decoy, mask_decoy, gaussian_coef=config.gaussian_coef), get_graph(Xcd, emb, proT5_emb, mask_crd_decoy, gaussian_coef=config.gaussian_coef), get_unfolded_graph(Xdu, emb_decoy, proT5_emb_decoy, mask_decoy, gaussian_coef=config.gaussian_coef)
    # Add cycle permutation
    Xcy1 = get_graph(Xcy1, cycle_emb1, proT5_cycle1, mask, gaussian_coef=config.gaussian_coef)
    Xcy2 = get_graph(Xcy2, cycle_emb2, proT5_cycle2, mask, gaussian_coef=config.gaussian_coef)
    # Xjf.requires_grad = True
    # create a batch of Xjf,Xkf,Xju,Xku,x_decoy
    Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2 = Xjf.unsqueeze(0),Xju.unsqueeze(0),Xd.unsqueeze(0), Xcd.unsqueeze(0), Xdu.unsqueeze(0),Xcy1.unsqueeze(0),Xcy2.unsqueeze(0)

    return Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2

def get_item_data(item_path):
    """Get item data from the item path"""
    # load data
    id = torch.load(item_path + '/id.pt')
    crd_backbone = torch.tensor(torch.load(item_path + '/crd_backbone.pt'),dtype=torch.get_default_dtype()) #backbone coordinates N,Calpha,C
    mask = torch.load(item_path + '/mask.pt')
    # change to 1,0 mask
    mask = torch.tensor(np.where(np.array(list(mask))=='+',1,0))
    # one hot encoding of the sequence
    seq_one_hot = torch.load(item_path + '/seq_one_hot.pt')
    seq = torch.load(item_path + '/seq.pt')
    proT5_emb = torch.load(item_path + '/proT5_emb.pt')
    # Add Cbeta atom to the coordinates
    crd_backbone = add_cb(crd_backbone)
    # Convert to angstrom
    crd_backbone = crd_backbone * C.NANO_TO_ANGSTROM 
    data = id, crd_backbone.unsqueeze(0), mask.unsqueeze(0), seq_one_hot.unsqueeze(0), seq, proT5_emb.unsqueeze(0)
    return data