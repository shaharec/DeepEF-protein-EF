import os
from tqdm import tqdm
import torch
import gc
import sidechainnet as scn
import numpy as np
n_sideChain = 14

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
def get_one_hot(seq):
  """get one hot from sequence"""
  seq_one_hot = torch.zeros((len(seq),20))
  for i,a in enumerate(seq):
    seq_one_hot[i][AA_MAP[a]] = 1
  return seq_one_hot


for thinning in [100]:#[30, 50, 70, 90, 95, 100]:
  path = './data/casp12_data_'+str(thinning)
  d = scn.load(casp_version=12, thinning=thinning)
  if(not(os.path.exists(path))):
    os.mkdir(path)
  f = open(f"{path}/metadata.txt", "a")
  f.write("Date: "+d['date']+'\n')
  f.write("settings: "+str(d['settings'])+'\n')
  f.write("description: "+str(d['description'])+'\n')
  f.close()
  for key in ['train', 'test', 'valid-10', 'valid-20', 'valid-30', 'valid-40', 'valid-50', 'valid-70', 'valid-90']:
    set_path = os.path.join(path,key)
    if(not(os.path.exists(set_path))):
      os.mkdir(set_path)
    for i in tqdm(range(len(d[key]['seq']))):
      id, seq, ang, crd, mask, sec, evo, res, ums, mod = d[key]['ids'][i], d[key]['seq'][i], d[key]['ang'][i], d[key]['crd'][i], d[key]['msk'][i], d[key]['sec'][i],d[key]['evo'][i] ,d[key]['res'][i], d[key]['ums'][i], d[key]['mod'][i]
      # create path
      protein_path = os.path.join(set_path,str(id))
      if(not(os.path.exists(protein_path))):
        os.mkdir(protein_path)
      #reshape coords
      crd_backbone = crd.reshape(len(seq),n_sideChain,-1)[:,:3,:] #backbone coordinates N,Calpha,C
      # save to files
      torch.save(id, protein_path+'/id.pt')
      torch.save(seq, protein_path+'/seq.pt')
      torch.save(get_one_hot(seq),protein_path+'/seq_one_hot.pt')
      torch.save(ang,protein_path+'/ang.pt')
      torch.save(crd,protein_path+'/crd.pt')
      torch.save(crd_backbone,protein_path+'/crd_backbone.pt')
      torch.save(mask,protein_path+'/mask.pt')
      torch.save(sec,protein_path+'/sec.pt')
      torch.save(evo,protein_path+'/evo.pt')
      gc.collect()
