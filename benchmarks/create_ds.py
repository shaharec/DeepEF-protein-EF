import pandas as pd
import os
import re
from tqdm import tqdm

# Function to split each segment
def split_segment(segment):
    match = re.match(r"([A-Za-z])(\d+)([A-Za-z])", segment)
    if match:
        return match.groups()
    return (segment)

# Function to handle multiple segments
def split_string(s):
    segments = s.split(':')
    split_segments = [split_segment(segment) for segment in segments]
    return split_segments

def create_blusom62_csv():
    """Create a csv file with blusom62 data."""
    pdb_dir = "./data/Processed_K50_dG_datasets/AlphaFold_model_PDBs/"
    mutation_dir ="./data/Processed_K50_dG_datasets/mutation_datasets/"
    blusom62_df = pd.DataFrame()
    # Get all data
    protein_list = os.listdir(mutation_dir)
    for protein in tqdm(protein_list):
        protein_df = pd.DataFrame(columns=["name","pdb_path", "mut_type", "deltaG"])
        protein_mutation = pd.read_csv(mutation_dir + protein )
        protein_df["name"] = protein_mutation['name']
        protein_df["pdb_path"] = pdb_dir + protein.replace(".csv", "")+".pdb"
        protein_df["mut_type"] = protein_mutation["mut_type"]
        protein_df["deltaG"] = protein_mutation["deltaG"]
        protein_df["Stabilizing_mut"] = protein_mutation["Stabilizing_mut"]
        # If there is no mutation, skip the protein
        if protein_mutation[protein_mutation['mut_type'] == 'wt'].empty:
            continue
        wt_deltaG = protein_mutation[protein_mutation['mut_type'] == 'wt']['deltaG'].iloc[0]
        protein_df['ddg'] = protein_df['deltaG'] - wt_deltaG
        # split the mut_type to tuples
        protein_df['mut_tuple'] = protein_df['mut_type'].apply(split_string)
        
        blusom62_df = pd.concat([blusom62_df, protein_df])
    blusom62_df.to_csv("./data/Processed_K50_dG_datasets/blusom62.csv", index=False)

def create_rosetta_csv():
    """Create a csv file with rosetta data."""
    pdb_dir = "./data/Processed_K50_dG_datasets/AlphaFold_model_PDBs/"
    mutation_dir ="./data/Processed_K50_dG_datasets/mutation_datasets/"
    rosetta_df = pd.DataFrame()
    # Get all data
    protein_list = os.listdir(mutation_dir)
    for protein in tqdm(protein_list):
        protein_df = pd.DataFrame(columns=["name","pdb_path", "mut_type", "deltaG"])
        protein_mutation = pd.read_csv(mutation_dir + protein )
        protein_df["name"] = protein_mutation['name']
        protein_df["pdb_path"] = pdb_dir + protein.replace(".csv", "")+".pdb"
        protein_df["mut_type"] = protein_mutation["mut_type"]
        protein_df["deltaG"] = protein_mutation["deltaG"]
        # split the mut_type to tuples
        protein_df['mut_tuple'] = protein_df['mut_type'].apply(split_string)
        
        rosetta_df = pd.concat([rosetta_df, protein_df])
    rosetta_df.to_csv("./data/Processed_K50_dG_datasets/rosetta.csv", index=False)    

def creat_foldx_ds():
    """Create a csv file with foldx data."""
    pdb_dir = "./data/Processed_K50_dG_datasets/AlphaFold_model_PDBs/"
    mutation_dir ="./data/Processed_K50_dG_datasets/mutation_datasets/"
    foldx_df = pd.DataFrame()
    # Get all data
    protein_list = os.listdir(mutation_dir)
    for protein in tqdm(protein_list):
        # create folder for each protein
        os.makedirs(f"./data/Processed_K50_dG_datasets/foldx/{protein.replace('.csv','')}", exist_ok=True)
        mutant_df = pd.DataFrame()
        protein_df = pd.DataFrame(columns=["name","pdb_path", "mut_type", "deltaG"])
        protein_mutation = pd.read_csv(mutation_dir + protein )
        # Check only for mutations
        protein_mutation["mut_special_type"] = protein_mutation['mut_type'].apply(lambda x: x[:3])
        filtered_mut = protein_mutation[(protein_mutation['mut_type'] != 'wt') & (protein_mutation['mut_special_type'] != "ins") & (protein_mutation['mut_special_type'] != "del")]
        if filtered_mut.empty:
            continue
        # Add wilde type sequence
        wt_seq = protein_mutation[protein_mutation['mut_type'] =='wt']['aa_seq'].iloc[0]
        wt_df = pd.DataFrame({'aa_seq': [wt_seq]}, index=[0])
        mutant_df['aa_seq'] = filtered_mut['aa_seq']
       # Concatenate wt_df with mutant_df, ensuring the wild type row is first
        mutant_df = pd.concat([wt_df, mutant_df]).reset_index(drop=True)
       
        # save protein df
        protein_df["name"] = filtered_mut['name']
        protein_df["pdb_path"] = pdb_dir + protein.replace(".csv", "")+".pdb"
        protein_df["mut_type"] = filtered_mut["mut_type"]
        protein_df["deltaG"] = filtered_mut["deltaG"]
    
        
        foldx_df = pd.concat([foldx_df, protein_df])
        protein_df.to_csv(f"./data/Processed_K50_dG_datasets/foldx/{protein.replace('.csv','')}/foldx.csv", index=False)
        #save mutant_df without header and as txt file
        mutant_df.to_csv(f"./data/Processed_K50_dG_datasets/foldx/{protein.replace('.csv','')}/mutant_file.txt", index=False, header=False, sep=' ')
    foldx_df.to_csv("./data/Processed_K50_dG_datasets/foldx_all.csv", index=False)

def main():
    # create_rosetta_csv()
    # creat_foldx_ds()
    create_blusom62_csv()
    

if __name__ == "__main__":
    main()

