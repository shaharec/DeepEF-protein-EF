import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from model.model_cfg import CFG
import gc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import constants as C
import pandas as pd


class SidChainDS(Dataset):
    """Protein dataset."""
    def __init__(self, data_path ,set_type,debug, LLM_EMB=True, outliners_path = None):
        """
            Initialize the dataset
        Args:
            data_path (str): data path
            set_type (str): 'train','test' or 'valid
            debug (bool): debug mode
            LLM_EMB (bool): use the large language model embeddings
            outliners_path (str): path to the outliners file'
        """
        self.data_path = data_path
        self.LLM_EMB = LLM_EMB
        self.pad_data = False
        self.set_type = set_type
        self.outliners_path = outliners_path
        self.data_dir = []
        if(set_type == 'valid'):
            self.folders  =[data_path+val_path for val_path in ['valid-10/','valid-20/','valid-30/','valid-40/','valid-50/']]
            for folder in self.folders:
                self.data_dir.extend([folder+f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))])
        else:
            self.data_dir = [os.path.join(data_path+set_type, f) for f in os.listdir(data_path+set_type) if os.path.isdir(os.path.join(data_path+set_type, f))]   
        if(debug):
            self.data_dir = self.data_dir[:CFG.debug_size]
        # remove the outliners
        if self.outliners_path:
            self.remove_outliners()
         # remove the mega-scale proteins 
        self.remove_megascale_proteins()
    
    def remove_outliners(self):
        """remove the outliners from the dataset"""
        outliners = pd.read_csv(self.outliners_path)
        outliners_ids = outliners['protein_id'].to_list()
        self.data_dir = [f for f in self.data_dir if not any(f_out in f for f_out in outliners_ids)]
    
    def remove_megascale_proteins(self):
        """remove the proteins from the mega-scale dataset"""
        mega_scale = pd.read_csv('./data/megascale_proteins.csv')
        mega_scale_ids = mega_scale['protein_name'].to_list()
        self.data_dir = [f for f in self.data_dir if not any(f_ms in f for f_ms in mega_scale_ids)]     
   
    def __getitem__(self, index):
        
        item_path = self.data_dir[index]
        
        decoy_path = self.data_dir[np.random.randint(len(self.data_dir))]
        while decoy_path == item_path:
            decoy_path = self.data_dir[np.random.randint(len(self.data_dir))]
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
        seq_one_hot = torch.load(item_path + '/seq_one_hot.pt')
        seq = torch.load(item_path + '/seq.pt')
        seq_decoy = torch.load(decoy_path + '/seq.pt')
        # proT5_emb = torch.zeros((len(seq),1024)) # for testing
        ang = torch.tensor(torch.load(item_path + '/ang.pt'))
        ang_backbone = torch.clone(ang)[:,:3] #angles for the backbone phi, psi, omega
        # Add Cbeta atom to the coordinates
        crd_backbone = self.add_cb(crd_backbone)
        crd_decoy = self.add_cb(crd_decoy)
        # Convert to angstrom
        crd_backbone = crd_backbone * C.NANO_TO_ANGSTROM 
        crd_decoy = crd_decoy * C.NANO_TO_ANGSTROM 
        
        # ProT5 embedding for protein mutation
        seq_mut =  torch.load(item_path + '/seq_mut.pt')
        # proT5_mut = torch.zeros((len(seq),1024)) # for testing
        # seq_mut = seq # for testing
        if self.LLM_EMB:
            proT5_mut = torch.load(item_path + '/proT5_emb_mut.pt')
            proT5_emb = torch.load(item_path + '/proT5_emb.pt')
            try:
                proT5_cycle1 = torch.load(item_path + '/proT5_emb_cycle1.pt')
            except:
                proT5_cycle1 = torch.load(item_path + '/proT5_emb_cycle.pt')
            proT5_cycle2 = torch.load(item_path + '/proT5_emb_cycle2.pt')
            proT5_cycle3 = torch.load(item_path + '/proT5_emb_cycle3.pt')
            proT5_cycle4 = torch.load(item_path + '/proT5_emb_cycle4.pt')
            proT5_cycle5 = torch.load(item_path + '/proT5_emb_cycle5.pt')
            proT5_cycle6 = torch.load(item_path + '/proT5_emb_cycle6.pt')
        else:
            proT5_emb = torch.zeros((len(seq),1024))
            proT5_mut = torch.zeros((len(seq),1024))
            proT5_cycle1 = torch.zeros((len(seq),1024))
            proT5_cycle2 = torch.zeros((len(seq),1024))
            proT5_cycle3 = torch.zeros((len(seq),1024))
            proT5_cycle4 = torch.zeros((len(seq),1024))
            proT5_cycle5 = torch.zeros((len(seq),1024))
            proT5_cycle6 = torch.zeros((len(seq),1024))
        
        if self.pad_data:
            data =  self.padding_data((id, crd_backbone, mask, seq_one_hot, seq,ang_backbone,ang, proT5_emb, proT5_mut,seq_mut, crd_decoy, mask_decoy, seq_decoy))
            return data
        
        return id, crd_backbone, mask, seq_one_hot, seq,ang_backbone, \
            ang, proT5_emb, proT5_mut,seq_mut, crd_decoy, mask_decoy, seq_decoy,proT5_cycle1,proT5_cycle2,\
            proT5_cycle3,proT5_cycle4,proT5_cycle5,proT5_cycle6

    def padding_data(self, data, max_len = CFG.seq_len):
        """
        Pad the data to the maximum length
        Args:
            data (tensor): data to pad
            max_len (int): maximum length to pad
        Returns:
            tensor: padded data
        """
        id, crd_backbone, mask, seq_one_hot, seq,ang_backbone, \
            ang, proT5_emb, proT5_mut,seq_mut, crd_decoy, mask_decoy, seq_decoy = data
        pad_len = max_len - crd_backbone.shape[0]
        crd_backbone = F.pad(crd_backbone,(0,0,0,0,0,pad_len))
        seq_one_hot = F.pad(seq_one_hot,(0,0,0,pad_len))
        proT5_emb = F.pad(proT5_emb,(0,0,0,pad_len))
        proT5_mut = F.pad(proT5_mut,(0,0,0,pad_len))
        mask = F.pad(mask,(0,pad_len))
        ang = F.pad(ang,(0,0,0,pad_len))
        ang_backbone = F.pad(ang_backbone,(0,0,0,pad_len))
        # decoy padding
        if crd_decoy.shape[0] > crd_backbone.shape[0]:
            crd_decoy = crd_decoy[:crd_backbone.shape[0],:,:]
            mask_decoy = mask[:crd_backbone.shape[0]]
        elif crd_decoy.shape[0] < crd_backbone.shape[0]:
            # add zeros to the end of the decoy
            pad_len = crd_backbone.shape[0] - crd_decoy.shape[0]
            crd_decoy = F.pad(crd_decoy,(0,0,0,0,0,pad_len))
            mask_decoy = F.pad(mask_decoy,(0,pad_len))

        
        return id, crd_backbone, mask, seq_one_hot, seq,ang_backbone, \
            ang, proT5_emb, proT5_mut,seq_mut, crd_decoy, mask_decoy, seq_decoy
            
    def __len__(self):
        return len(self.data_dir)

    def get_dist_matrix(self,X):
        """
        Return the node distence matrix
        Args:
            X (tensor):X embeded [n_nodes ,num_atoms=4,new_cords_size]
        Returns:
            tensor : [n_nodes,n_nodes ,atom_dist=16] tensor
        """
        N_residu,N_atoms,coords_size = X.shape
        X = X.reshape(N_residu*N_atoms,coords_size)
        D = torch.cdist(X,X,p=2)
        D = D.reshape(N_residu,N_atoms,N_residu,N_atoms)
        D = torch.swapaxes(D,1,2)
        D = D.reshape(N_residu,N_residu,N_atoms*N_atoms)
        return D
  
    def add_cb(self,crd_coords):
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
    
class PEFDataset(Dataset):
    '''
    Deep energy function dataset.
    Data item structure:
        # 3D coordinates of the protein
            * coordsAlpha
            * coordsBeta 
            * coordsC
            * coordsCa
            * coordsN
            * coordsAlpha native
            * coordsBeta native
            * coordsC native
            * coordsCa native
            * coordsN native
        # Embeddings
           * Large languege model embeddings
           * hand selected features
    '''

    def __init__(self, file_df, datapath=CFG.data_path, constraint=True, homothresh=CFG.homothresh, type='train',
                 train_type='RoseTTAFold'):
        """_summary_
        Data set for the Deep energy function dataset.
        Args:
            file_df (list): _description_.files dataframe from the data path
            datapath (string,): _description_. Defaults to CFG.data_path.
            homothresh (float, optional): _description_. Defaults to CFG.homothresh.
            type (str, optional): _description_. Defaults to 'train'.
        """
        self.datapath = datapath
        self.filenames = file_df.copy()
        self.homothresh = homothresh
        self.type = type
        self.train_type = train_type
        # remove the files with homology greater than homothresh
        if type == 'train' and constraint:
            self.check_data_constraint()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        index_path = os.path.join(self.datapath, self.filenames[index])
        # Sequence of the protein
        seq = torch.load(os.path.join(index_path, 'seq.pt'))
        seq_decoy = seq[:, torch.randperm(seq.shape[1])]
        id = torch.load(os.path.join(index_path, 'ids.pt'))  # string id
        # 3D coordinates of the protein
        coordsAlpha = torch.load(os.path.join(index_path, 'CoordAlpha.pt'))
        coordsBeta = torch.load(os.path.join(index_path, 'CoordBeta.pt'))
        coordsC = torch.load(os.path.join(index_path, 'CoordC.pt'))
        coordsN = torch.load(os.path.join(index_path, 'CoordN.pt'))
        # 3D coordinates of the protein native
        coordsAlpha_native = torch.load(os.path.join(index_path, 'CoordCaNative.pt'))
        coordsBeta_native = torch.load(os.path.join(index_path, 'CoordCbNative.pt'))
        coordsC_native = torch.load(os.path.join(index_path, 'CoordCNative.pt'))
        coordsN_native = torch.load(os.path.join(index_path, 'CoordNNative.pt'))
        # Check glycine value
        glyIndices = torch.where(coordsBeta_native[:, 0] > 5e4)[0]
        if torch.any(glyIndices):
            coordsBeta_native[glyIndices, :] = self.get_c_beta(coordsN_native[glyIndices, :],
                                                          coordsAlpha_native[glyIndices, :],
                                                          coordsC_native[glyIndices, :])
        # Check glycine value decoy
        glyIndices = torch.where(coordsBeta[:, 0] > 5e4)[0]
        if torch.any(glyIndices):
            coordsBeta[glyIndices, :] = self.get_c_beta(coordsN[glyIndices, :], coordsAlpha[glyIndices, :],
                                                   coordsC[glyIndices, :])
        # Masks
        mask = torch.load(os.path.join(index_path, 'mask.pt'))
        nativemask = torch.load(os.path.join(index_path, 'nativemask.pt'))
        # Embeddings
        n_nodes = coordsAlpha.shape[0]
        esm_embed = torch.stack(torch.load(os.path.join(index_path, 'emb_esm.pt')))[0][:10]
        esm_embed = esm_embed.repeat(n_nodes, 1)
        # Concatenate the coordinates
        Xd = self.concat_coords(coordsAlpha, coordsBeta, coordsC, coordsN)
        Xn = self.concat_coords(coordsAlpha_native, coordsBeta_native, coordsC_native, coordsN_native)

        # Convert nano to angstrom
        Xd = Xd * C.NANO_TO_ANGSTROM
        Xn = Xn * C.NANO_TO_ANGSTROM

        return seq, seq_decoy, id, Xd, Xn, mask, nativemask, esm_embed

    def read_protein(self, index):
        """
        Read the protein data from the index path

        Args:
            index (int): index of protein in the dataset
        """
        index_path = os.path.join(self.datapath, self.filenames[index])
        # Sequence of the protein
        seq = torch.load(os.path.join(index_path, 'seq.pt')).to(CFG.device)
        id = torch.load(os.path.join(index_path, 'ids.pt'))  # string id
        # 3D coordinates of the protein
        coordsAlpha = torch.load(os.path.join(index_path, 'CoordAlpha.pt')).to(CFG.device)
        coordsBeta = torch.load(os.path.join(index_path, 'CoordBeta.pt')).to(CFG.device)
        coordsC = torch.load(os.path.join(index_path, 'CoordC.pt')).to(CFG.device)
        coordsN = torch.load(os.path.join(index_path, 'CoordN.pt')).to(CFG.device)
        # 3D coordinates of the protein native
        coordsAlpha_native = torch.load(os.path.join(index_path, 'CoordCaNative.pt')).to(CFG.device)
        coordsBeta_native = torch.load(os.path.join(index_path, 'CoordCbNative.pt')).to(CFG.device)
        coordsC_native = torch.load(os.path.join(index_path, 'CoordCNative.pt')).to(CFG.device)
        coordsN_native = torch.load(os.path.join(index_path, 'CoordNNative.pt')).to(CFG.device)
        # Masks
        mask = torch.load(os.path.join(index_path, 'mask.pt')).to(CFG.device)
        nativemask = torch.load(os.path.join(index_path, 'nativemask.pt')).to(CFG.device)
        # Embeddings
        esm_embed = torch.load(os.path.join(index_path, 'emb_esm.pt'))[0].to(CFG.device)

        return seq, id, coordsAlpha, coordsBeta, coordsC, coordsN, coordsAlpha_native, coordsBeta_native, coordsC_native, coordsN_native, mask, nativemask, esm_embed

    def concat_coords(self, coordsAlpha, coordsBeta, coordsC, coordsN):
        """
        Concatenate the coordinates
        output: X (torch.tensor): concatenated coordinates [N,4,3]
        """
        coordsAlpha, coordsBeta, coordsC, coordsN = coordsAlpha.unsqueeze(1), coordsBeta.unsqueeze(
            1), coordsC.unsqueeze(1), coordsN.unsqueeze(1)
        X = torch.cat((coordsAlpha, coordsBeta, coordsC, coordsN), dim=1)
        return X

    def check_data_constraint(self):
        """
        Check the data constraint and remove the files with homology greater than homothresh.
        Check mask and native mask.
        Update file names list.
        """
        print('Checking data constraint...')
        new_filenames = []
        for i in tqdm(range(len(self.filenames))):
            (seq, id, coordAlpha, coordBeta, coordC, coordN, coordAlphaNative,
             coordBetaNative, coordCNative, coordNNative, mask, nativemask, embedding) = self.read_protein(i)
            dt = torch.get_default_dtype()
            coordN = coordN.to(dt)
            coordAlpha = coordAlpha.to(dt)
            coordC = coordC.to(dt)
            coordBeta = coordBeta.to(dt)
            seq = seq.to(dt)
            embedding = embedding.to(dt)
            coordNNative = coordNNative.to(dt)
            coordAlphaNative = coordAlphaNative.to(dt)
            coordCNative = coordCNative.to(dt)
            coordBetaNative = coordBetaNative.to(dt)

            s = seq.mean(-1)
            if (self.homothresh is not None) and (s.max() > self.homothresh):
                # print("protein is too homogeneous", id)
                continue

            if (self.train_type is not None) and (not self.train_type in id):
                continue

            
            # scale = 1e-2
            # Mnat = nativemask
            # M = msk & Mnat

            # ind = torch.where(M)[0]
            # istart = ind[0]
            # ilast = ind[-1]
            # M = M[istart:ilast + 1]
            # msk = msk[istart:ilast + 1]
            # msk = msk.type('torch.FloatTensor')
            # if torch.any(msk == 0):
            #     # print("id problem", id)
            #     idx = (idx + 1) % self.__len__()
            #     continue

            new_filenames.append(self.filenames[i])

            torch.cuda.empty_cache()
            gc.collect()

        self.filenames = new_filenames

    @staticmethod
    def get_c_beta(N, CA, C):
        # CB = CA + c1*(N-CA) + c2*(C-CA) + c3* (N-CA)x(C-CA)
        dt = torch.get_default_dtype()
        N = N.to(dt)
        CA = CA.to(dt)
        C = C.to(dt)

        CAmN = N - CA
        # CAmN = CAmN / torch.sqrt(CAmN ** 2).sum(dim=2, keepdim=True)
        CAmC = C - CA
        # CAmC = CAmC / torch.sqrt(CAmC ** 2).sum(dim=2, keepdim=True)
        ANxAC = torch.cross(CAmN, CAmC, dim=1)

        A = torch.cat((CAmN.reshape(-1, 1), CAmC.reshape(-1, 1), ANxAC.reshape(-1, 1)), dim=1)
        c = torch.tensor([0.5507, 0.5354, -0.5691]) / 100  # torch.tensor([1.1930, 1.2106, -2.7906]) #
        b = (A @ c).reshape(-1, 3)
        CB = CA - b

        return CB


def fetch_dataloader(data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    # Sidechainnet dataset
    if params.dataset == 'scn':
        train_loader= DataLoader(SidChainDS(data_path=data_dir,set_type='train', debug=params.debug, LLM_EMB = params.LLM_EMB,outliners_path = params.outliners_path), batch_size=params.batch_size, shuffle=True,
                                            num_workers=params.num_workers,
                                            pin_memory=params.cuda)
        valid_loader= DataLoader(SidChainDS(data_path=data_dir,set_type='valid', debug=params.debug, LLM_EMB = params.LLM_EMB), batch_size=params.batch_size, shuffle=True,
                                            num_workers=params.num_workers,
                                            pin_memory=params.cuda)

        test_loader= DataLoader(SidChainDS(data_path=data_dir,set_type='test', debug=params.debug, LLM_EMB = params.LLM_EMB), batch_size=params.batch_size, shuffle=True,
                                            num_workers=params.num_workers,
                                            pin_memory=params.cuda)
    else:
        # Get the filenames from the train folder
        file_names = os.listdir(data_dir)
        if params.debug:
            file_names = file_names[:CFG.debug_size]
        # Split the data into train, validation and test set
        X_train, X_rem, y_train, y_rem = train_test_split(file_names,file_names, train_size=CFG.split_train_size,
                                                        random_state=CFG.seed)
        # Now since we want the valid and test size to be equal (10% each of overall data). 
        # we have to define valid_size=0.5 (that is 50% of remaining data)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)
        # Now we have the data split in training, validation and test set
        train_loader= DataLoader(PEFDataset(X_train,datapath=data_dir,constraint = params.constraint), batch_size=params.batch_size, shuffle=True,
                                            num_workers=params.num_workers,
                                            pin_memory=params.cuda)
        valid_loader= DataLoader(PEFDataset(X_valid,datapath=data_dir,constraint = params.constraint), batch_size=params.batch_size, shuffle=True,
                                            num_workers=params.num_workers,
                                            pin_memory=params.cuda)

        test_loader= DataLoader(PEFDataset(X_test,datapath=data_dir,constraint = params.constraint), batch_size=params.batch_size, shuffle=True,
                                            num_workers=params.num_workers,
                                            pin_memory=params.cuda)
    return train_loader, valid_loader, test_loader

def fetch_inference_loader(data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        data_dir (str): inferece data directory
        params (params class): hyperparameters
    """
    inf_file_names = os.listdir(data_dir)
    amino_inference_loader = DataLoader(PEFDataset(inf_file_names,datapath=data_dir,constraint = params.constraint), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
    return amino_inference_loader


class params:
    def __init__(self,batch_size,num_workers,cuda,constraint, dataset,debug=False,LLM_EMB=True,outliners_path=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cuda = cuda
        self.debug = debug
        self.constraint = constraint
        self.dataset = dataset
        self.LLM_EMB =LLM_EMB
        self.outliners_path = outliners_path
        
