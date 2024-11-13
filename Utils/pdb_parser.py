# PDB parser
import torch
from Utils.train_utils import add_cb
import re
import constants as C
from transformers import T5Tokenizer, T5EncoderModel

# Define the list of atom types to save
atom_types = ['N', 'CA', 'C']
MISSING_COORD = torch.tensor([0, 0, 0])
OTHER_ACID = '-'
# Convert three-letter amino acid codes to one-letter codes
aa_dict = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

def extract_pdb_info(pdb_file, atom_types=['N', 'CA', 'C'], chain_id='A'):
    coordinates = {}
    sequence = {}
    
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                splited_line = line.split()
                atom_id = splited_line[1]
                atom_type = splited_line[2]
                residue_type = splited_line[3]
                chain_line_id = splited_line[4]
                if chain_line_id not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    # print(splited_line)
                    chain_line_id = splited_line[4][0]
                    residue_id = splited_line[4][1:]
                    x = float(splited_line[5])
                    y = float(splited_line[6])
                    z = float(splited_line[7])
                else:
                    residue_id = splited_line[5]
                    x = float(splited_line[6])
                    y = float(splited_line[7])
                    z = float(splited_line[8])
                
                if chain_line_id != chain_id:
                    continue
                # Extract coordinates
                if atom_type not in atom_types:
                    continue
                
                if residue_id not in coordinates.keys():
                    coordinates[residue_id] = {}
                    for atom in atom_types:
                        coordinates[residue_id][atom] = [0.0, 0.0, 0.0]
                
                coordinates[residue_id][atom_type] = [x, y, z]
                
                # Extract amino acid for sequence
                if residue_type not in aa_dict.keys():
                    sequence[residue_id] = 'X' 
                else:
                    sequence [residue_id] = aa_dict[residue_type]
    
    sequence_str = ''.join([''.join([v for k, v in sequence.items()])])
    coordinates = torch.stack([torch.stack([torch.tensor(v) for v in list(v.values())]) for v in coordinates.values()])
        
    return coordinates, sequence_str, sequence

def get_pdb_data(pdb_file_path,chain_id):
    """Get the coordinates and sequence of a protein from a pdb file"""
    protein_name = re.search(r'\/(\w+).pdb', pdb_file_path).group(1)
    coords, sequence, seq_dict =extract_pdb_info(pdb_file_path, chain_id= chain_id)
    coords = torch.tensor(coords)
    # add CB
    coords = add_cb(coords)
    # Add nano to angstrom conversion
    coords = coords * C.NANO_TO_ANGSTROM
    
    mask_tensor = (coords != 0).all(dim=2).all(dim=1).float()

    proT5_emb = get_emb([sequence])
    
    if len(sequence) != coords.shape[0]:
        print('Sequence length does not match the number of residues')
        print('Protein:', protein_name)
        print('Sequence length:', len(sequence))
        print('Number of residues:', coords.shape[0])
        
    data = {"coords": coords, "sequence": sequence, "mask_tensor": mask_tensor, "proT5_emb": proT5_emb}
    return data
    
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))


def get_emb(sequence_examples):
    #@title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
    # Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50 in half-precision) 
    transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print("Loading: {}".format(transformer_link))
    model = T5EncoderModel.from_pretrained(transformer_link)
    model.full() if device=='cpu' else model.half() # only cast to full-precision if no GPU is available
    model = model.to(device)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False )
    
    # get the sequence embeddings
    seq_len = len(sequence_examples[0])
    # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
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
    return emb_0

