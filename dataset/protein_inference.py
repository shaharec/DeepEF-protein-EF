from model.data_loader import fetch_dataloader,fetch_inference_loader
from model.data_loader import params as data_params
from model.model_cfg import CFG
# from model.net import ProteinEnergyNet
from model.hydro_net import PEM
from Utils.train_utils import *
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import gc
import time
import sys
import glob
import pandas as pd
import wandb
import constants as C
from transformers import T5Tokenizer, T5EncoderModel
import re
from Bio.PDB import PDBParser

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_DIR = glob.glob('./data/casp12_data_100/train/*')

def get_protein_emb(seq):
    """
    Get the protein embedding from the proT5 model
    """
    # @title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
    # Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50 in half-precision) 
    transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print("Loading: {}".format(transformer_link))
    model = T5EncoderModel.from_pretrained(transformer_link)
    model.full() if device=='cpu' else model.half() # only cast to full-precision if no GPU is available
    model = model.to(device)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False )

    # sequence_examples = ["PRTEINO", "GPSGLGLPAGLYAFNSGGISLDLGINDPVPFNTVGSQFGTAISQLDADTFVISETGFYKITVIANTATASVLGGLTIQVNGVPVPGTGSSLISLGAPIVIQAITQITTTPSLVEVIVTGLGLSLALGTSASIIIEKVA"]
    sequence_examples = [seq]
    seq_len = len(sequence_examples[0])
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)


    # extract embeddings for the first ([0,:]) sequence in the batch while removing padded & special tokens ([0,:7]) 
    emb_0 = embedding_repr.last_hidden_state[0,:seq_len]
    # move embeddings to cpu and cast to full-precision tensor
    emb_0 =torch.tensor(emb_0,dtype=torch.float32)
    return emb_0

def get_protein_data(item_path, data_dir = DATA_DIR,new_seq= None):
    """
    Get the data for the protein inference, if there is a new sequence, it will be used for the inference
    """
    decoy_path = data_dir[np.random.randint(len(data_dir))]
    while decoy_path == item_path:
        decoy_path = data_dir[np.random.randint(len(data_dir))]
    # load data
    id = torch.load(item_path + '/id.pt')
    crd_backbone = torch.tensor(torch.load(item_path + '/crd_backbone.pt'),dtype=torch.get_default_dtype()) #backbone coordinates N,Calpha,C
    crd_decoy = torch.tensor(torch.load(decoy_path + '/crd_backbone.pt'),dtype=torch.get_default_dtype()) #backbone coordinates N,Calpha,C

    mask = torch.load(item_path + '/mask.pt')
    mask_decoy = torch.load(decoy_path + '/mask.pt')
    # change to 1,0 mask
    mask = torch.tensor(np.where(np.array(list(mask))=='+',1,0))
    mask_decoy = torch.tensor(np.where(np.array(list(mask_decoy))=='+',1,0))
    # one hot encoding of the sequence
    if new_seq:
        seq = new_seq
        seq_one_hot = get_one_hot(seq)
        proT5_emb = get_protein_emb(seq)
    else:
        seq = torch.load(item_path + '/seq.pt')
        seq_one_hot = torch.load(item_path + '/seq_one_hot.pt')
        proT5_emb = torch.load(item_path + '/proT5_emb.pt')

    seq_decoy = torch.load(decoy_path + '/seq.pt')
    # proT5_emb = torch.zeros((len(seq),1024)) # for testing
    ang = torch.tensor(torch.load(item_path + '/ang.pt'))
    ang_backbone = torch.clone(ang)[:,:3] #angles for the backbone phi, psi, omega
    # Add Cbeta atom to the coordinates
    crd_backbone = add_cb(crd_backbone)
    crd_decoy = add_cb(crd_decoy)
    # Convert to angstrom
    crd_backbone = crd_backbone * C.NANO_TO_ANGSTROM 
    crd_decoy = crd_decoy * C.NANO_TO_ANGSTROM 

    # ProT5 embedding for protein mutation
    proT5_mut = torch.load(item_path + '/proT5_emb_mut.pt')
    seq_mut =  torch.load(item_path + '/seq_mut.pt')
    # proT5_mut = torch.zeros((len(seq),1024)) # for testing
    # seq_mut = seq # for testing
    # unsqueeze the data and save it in a list
    data = id, crd_backbone.unsqueeze(0), mask.unsqueeze(0), seq_one_hot.unsqueeze(0), seq,ang_backbone.unsqueeze(0),ang.unsqueeze(0), proT5_emb.unsqueeze(0), proT5_mut.unsqueeze(0),[seq_mut],\
        crd_decoy.unsqueeze(0), mask_decoy.unsqueeze(0), [seq_decoy]
    return data

def get_model_input(data):
    """Get the model input from the data"""
    id, crd_backbone, mask, seq_one_hot, seq,ang_backbone, ang,\
                proT5_emb, proT5_mut,seq_mut, crd_decoy, mask_crd_decoy, seq_crd_decoy = data
            
    # wild type and mutant type
    Xjf = crd_backbone.to(device) # wilde type structure folded
    Xkf = torch.clone(Xjf).to(device) # mutant structure folded
    Xju = torch.clone(Xjf).to(device) # wilde type structure unfolded
    Xku = torch.clone(Xjf).to(device) # mutant structure unfolded
    Xcd = crd_decoy.to(device) # decoy structure

    # change the decoy len to match the native len
    if Xcd.shape[1] > Xjf.shape[1]:
        Xcd = Xcd[:,:Xjf.shape[1],:,:]
        mask_crd_decoy = mask_crd_decoy[:,:Xjf.shape[1]]
    elif Xcd.shape[1] < Xjf.shape[1]:
        #   add zeros to the end of the decoy
        Xcd = torch.cat((Xcd, torch.zeros(Xjf.shape[0],Xjf.shape[1] - Xcd.shape[1], *Xcd.shape[2:]).to(device)), dim=1)
        mask_crd_decoy = torch.cat((mask_crd_decoy, torch.zeros(mask.shape[0],mask.shape[1] - mask_crd_decoy.shape[1])), dim=1)


    # native structure and decoy structure
    Xd = torch.clone(Xjf).to(device)
    seq_one_hot = seq_one_hot.to(device) # [batch_size,20,seq_len]

    # create decoy sequence
    seq_decoy,mask_decoy, proT5_emb_decoy = mix_A_acid(seq_one_hot = seq_one_hot, emb=proT5_emb, mask = mask,val_type='train',device=device)

    #emb = torch.cat((esm_embed,seq),dim=2)
    emb = seq_one_hot.to(device)
    emb_decoy = seq_decoy.to(device)
    emb_mut = get_one_hot(seq_mut[0]).to(device)

    # move proT5_emb to device
    proT5_mut, proT5_emb_decoy, proT5_emb = proT5_mut.to(device), proT5_emb_decoy.to(device), proT5_emb.to(device)

    # squeeze the data
    Xd, Xjf, Xkf, Xju, Xku, Xcd = Xd.squeeze(), Xjf.squeeze(), Xkf.squeeze(), Xju.squeeze(), Xku.squeeze(), Xcd.squeeze()
    emb_decoy, emb, emb_mut = emb_decoy.squeeze(), emb.squeeze(), emb_mut.squeeze()
    mask_decoy, mask, mask_crd_decoy= mask_decoy.squeeze(), mask.squeeze(), mask_crd_decoy.squeeze()
    proT5_emb_decoy, proT5_emb, proT5_mut = proT5_emb_decoy.squeeze(), proT5_emb.squeeze(), proT5_mut.squeeze()

    # get folded graph  
    Xjf,Xkf = get_graph(Xjf, emb, proT5_emb, mask), get_graph(Xkf, emb_mut, proT5_mut, mask)
    # get unfolded graph
    Xju,Xku = get_unfolded_graph(Xju, emb, proT5_emb, mask), get_unfolded_graph(Xku, emb_mut, proT5_mut, mask)
    # get decoy graph
    Xd, Xcd = get_graph(Xd, emb_decoy, proT5_emb_decoy, mask_decoy), get_graph(Xcd, emb, proT5_emb, mask_crd_decoy)
    # Xjf.requires_grad = True
    # create a batch of Xjf,Xkf,Xju,Xku,x_decoy
    Xjf,Xkf,Xju,Xku,Xd,Xcd = Xjf.unsqueeze(0),Xkf.unsqueeze(0),Xju.unsqueeze(0),Xku.unsqueeze(0),Xd.unsqueeze(0), Xcd.unsqueeze(0)    
    X = torch.cat((Xjf,Xkf,Xju,Xku,Xd,Xcd),dim=0)
    
    return X

def get_model(model_path="./res/trianed_models_energyLimit-inst_norm/3_final_model.pt"):
    """Get the model"""
    model = PEM(layers=CFG.num_layers,gaussian_coef=CFG.gaussian_coef).to(CFG.device)
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
# optimizer = optim.SGD(model.parameters(), lr=CFG.lr)
    model,optimizer,epoch,loss,valid_loss = load_checkpoint(model_path,model,optimizer,device=CFG.device)
    return model

def print_statistics(E):
    """Print the statistics of the energy"""
    Ejf, Ekf, Eju, Eku, Exd, Ecd = E[0], E[1], E[2], E[3], E[4], E[5]
    print(f'Energy of the native structure: {Ejf:.3f}')
    print(f'Energy of the mutant structure: {Ekf:.3f}')
    print(f'Energy of the unfolded native structure: {Eju:.3f}')
    print(f'Energy of the unfolded mutant structure: {Eku:.3f}')
    print(f'Energy of the decoy structure: {Exd:.3f}')
    print(f'Energy of the decoy structure: {Ecd:.3f}')
    # print energy differences 
    print(f'DeltaG native and unfolded native structure: {Eju-Ejf:.3f}')
    print(f'DeltaG mutant and unfolded mutant structure: {Eku-Ekf:.3f}')
    print(f'Energy difference between native and decoy structure: {Exd-Ejf:.3f}')
    print(f'Energy difference between native and coords decoy structure: {Ecd-Ejf:.3f}')
    
    

def inference_protein():
    """Inference the protein"""
    item_path = './data/casp12_data_100/train/1A0N_2_B'
    new_seq = 'GSTGVTLFVASYDYEARTEDDLSFHKGEKFQILNSSEGDWWEARSLTTGETGYIPSNYVAPVDSIQAEE'
    # get the data
    print(f'Getting the data of the protein {item_path.split("/")[-1]}')
    data = get_protein_data(item_path,new_seq = new_seq)
    # get the model input
    print(f'Getting the model input of the protein {item_path.split("/")[-1]}')
    X = get_model_input(data)
    X_msc = torch.load("./data/MegaScale/folded_A01N.pt")
    # replace embeddings 
    X[0,4:-7,-1044:-20] = X_msc[0,:,-1044:-20]
    # load the model
    print(f'Load the model')
    model = get_model()
    model.eval()
    # inference
    print(f'Inference the protein {item_path.split("/")[-1]}')
    with torch.no_grad():
        E = model(X)
    print_statistics(E)
    return E

def get_coords(pdb_file_path, protein_name):
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure(protein_name, pdb_file_path)
    out_data = dict(dict())
    sequence_dict = dict()
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_id = residue.id[1] - 1
                atom_data_dict = {atom_type: [] for atom_type in atom_types}
                if 'CA' in residue:
                    for atom in residue:
                        # Check if the atom type is in the list of atom types to save
                        if atom.get_name() in atom_types:
                            # Get the coordinates of the atom
                            coord = atom.get_coord()
                            # Add the coordinates to the corresponding atom type list in the data dictionary
                            atom_data_dict[atom.get_name()] = torch.from_numpy(coord)
                    if 'C' not in residue:
                        atom_data_dict['C'] = MISSING_COORD
                    if 'N' not in residue:
                        atom_data_dict['N'] = MISSING_COORD
                    if 'CB' not in residue and residue.get_resname() == 'GLY':
                        try:
                            atom_data_dict['CB'] = torch.from_numpy(residue['2HA'].get_coord())
                        except KeyError:
                            atom_data_dict['CB'] = torch.from_numpy(residue['HA2'].get_coord())
                        # atom_data_dict['CB'] = MISSING_COORD
                    # store the atoms foreach residue id, id is stored for mask later
                    sequence_dict[residue_id] = atom_data_dict
    return torch.stack([torch.stack(list(v.values())) for v in sequence_dict.values()])


def compar_coords(pdb_file1,pdb_file2):
    # compare the coordinates of the pdb files
    protein_name = 'A01N'
    coords1 = get_coords(pdb_file1,protein_name)
    coords2 = get_coords(pdb_file2,protein_name)
    print ("finished coords")
    
if __name__ == "__main__":
    inference_protein()
    # compar_coords("./data/MegaScale/1A0N-AlphaFold.pdb","./data/MegaScale/1A0N-databank.pdb")