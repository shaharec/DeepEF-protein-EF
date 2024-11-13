import pyrosetta
pyrosetta.init()

import sys
from pyrosetta import pose_from_pdb
from pyrosetta.toolbox.mutants import mutate_residue
from dataset import benckmark_datasets
import re
from tqdm import tqdm
import pandas as pd
# Initialize PyRosetta
pyrosetta.init()

# Function to split each segment
def split_segment(segment):
    match = re.match(r"([A-Za-z])(\d+)([A-Za-z])", segment)
    if match:
        return match.groups()
    return (segment)

def split_string(s):
    segments = s.split(':')
    split_segments = [split_segment(segment) for segment in segments]
    return split_segments

def calculate_delta_g(pdb_file, mutations):
    # Load the PDB file
    pose = pose_from_pdb(pdb_file)
    
    # Set up the score function
    scorefxn = pyrosetta.get_fa_scorefxn()
    
    # Calculate the initial free energy
    initial_score = scorefxn(pose)
    
    # Apply mutations
    for old_amino_acif, residue_number, new_amino_acid in mutations:
        mutate_residue(pose, int(residue_number), new_amino_acid)
    
    # Calculate the final free energy
    final_score = scorefxn(pose)
    
    # Compute ΔG
    delta_g = final_score - initial_score
    
    return initial_score, final_score, delta_g


def test_calculate_delta_g():
   # List of PDB files and their mutations
    base_dir = "./data/Processed_K50_dG_datasets/AlphaFold_model_PDBs/"
    pdb_files = [base_dir + "1A0N.pdb"]
    mutations_dict = {
        "A10N.pdb": [(10, 'P')]
    }

    results = []

    for pdb_file in pdb_files:
        mutations = mutations_dict.get(pdb_file, [])
        initial_score, final_score, delta_g = calculate_delta_g(pdb_file, mutations)
        results.append((pdb_file, initial_score, final_score, delta_g))
        print(f"Protein: {pdb_file}, Initial Score: {initial_score}, Final Score: {final_score}, ΔG: {delta_g}")

def validate_deltaG(quarter):
    dataset = benckmark_datasets("Rosetta")
    df_results = pd.DataFrame(columns=["name","pdb_path", "mut_type", "rosetta_wildtype", "rosetta_mutant", "rosetta_deltaG", "deltaG"])
    
    # Calculate start and end indices for this quarter
    total_len = len(dataset.dataset)
    quarter_size = total_len // 4
    start_idx = (quarter - 1) * quarter_size
    end_idx = start_idx + quarter_size if quarter < 4 else total_len
    
    for i in tqdm(range(start_idx, end_idx)):
        item = dataset.get_item(i)
        pdb_file = item['pdb_path']
        # Check only for mutations
        if (item['mut_type'] != 'wt' and item['mut_type'][:3] != "ins" and item['mut_type'][:3] != "del"):
            mutations = split_string(item['mut_type'])
            initial_score, final_score, delta_g = calculate_delta_g(pdb_file, mutations)
            df_results.loc[i] = [item['name'], item['pdb_path'], item['mut_type'], initial_score, final_score, delta_g, item['deltaG']]
            if(delta_g != 0):
                print(f"Protein: {pdb_file}, Initial Score: {initial_score}, Final Score: {final_score},  RosettaΔΔG: {delta_g}, ΔG: {item['deltaG']}")
   
        df_results.to_csv("./data/Processed_K50_dG_datasets/rosetta_valid.csv", index=False)
    # Add correlation metrics
    df_results["pearson"] = df_results["rosetta_deltaG"].corr(df_results["deltaG"], method='pearson')
    df_results["spearman"] = df_results["rosetta_deltaG"].corr(df_results["deltaG"], method='spearman')
    df_results.to_csv(f"./data/Processed_K50_dG_datasets/rosetta_valid_{quarter}.csv", index=False)

if __name__ == "__main__":
    # test_calculate_delta_g()asdasddas
    # get from parameters the data quarter
    quarter = int(sys.argv[1])
    print(f"quarter number{quarter}") 
    validate_deltaG(quarter)