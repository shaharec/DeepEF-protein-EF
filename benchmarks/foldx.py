import os
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import re
import pandas as pd
import sys

# Path to FoldX executable
foldx_path = "/cs/casp15/Shahar/foldx/foldx_20241231"
base_path = '/cs/casp15/Shahar/DeepPEF'
DEBUG = False

def run_foldx(protein, foldx_ds_path, pdb_path):
    # Check if mutant file is empty
    mutant_file = f'{foldx_ds_path}/{protein}/mutant_file.txt'
    # Check if mutant file exists and not empty
    if not os.path.exists(mutant_file):
        return
    if os.path.getsize(mutant_file) == 0:
        return
    # Create a directory for FoldX output
    os.makedirs(f'{foldx_ds_path}/{protein}/foldx_output2', exist_ok=True)
    output_dir = f'{foldx_ds_path}/{protein}/foldx_output2'
    # Path to the original PDB file
    pdb_name = f'{protein}.pdb'
    # Mutation path
    mutation_path = f'{foldx_ds_path}/{protein}/mutant_file.txt'
    # Run FoldX stability analysis
    foldx_command = f"{foldx_path} --command=BuildModel --pdb-dir={pdb_path} --pdb={pdb_name} --mutant-file={mutation_path} --output-dir={output_dir}"
    subprocess.run(foldx_command, shell=True)
    print(f"Finished {protein}")

def validate_foldx(fraction):
    foldx_ds_path = base_path + "/data/Processed_K50_dG_datasets/foldx"
    pdb_path = base_path + "/data/Processed_K50_dG_datasets/AlphaFold_model_PDBs"
    foldx_ds = os.listdir(foldx_ds_path)
    
    # Calculate start and end indices for this quarter
    total_len = len(foldx_ds)
    fraction_size = total_len // 10
    start_idx = (fraction - 1) * fraction_size
    end_idx = start_idx + fraction_size if fraction < 10 else total_len
    
    if DEBUG:
        foldx_ds = foldx_ds[:5]
    else:
        foldx_ds = foldx_ds[start_idx:end_idx]
        
    for protein in tqdm(foldx_ds):
        print(f"Running FoldX for {protein}")
        run_foldx(protein, foldx_ds_path, pdb_path)
    
    print("Quarter {} is done".format(fraction))

def test_foldx():
    # Path to FoldX executable
    foldx_path = "/cs/casp15/Shahar/foldx/foldx_20241231"

    # Path to the original PDB file
    pdb_path = "./data/Processed_K50_dG_datasets/AlphaFold_model_PDBs"
    pdb_name = '1A32.pdb'

    # Create a directory for FoldX output
    output_dir = "./data/foldx_output2"
    os.makedirs(output_dir, exist_ok=True)
    # Mutation path
    mutation_path = './benchmarks/mutant_file.txt'
    wild_type = 'SPEVQIAILTEQINNLNEHLRVHKKDHHSRRGLLKMVGKRRRLLAYLRNKDVARYREIVEKLG'

    # Run FoldX stability analysis
    foldx_command = f"{foldx_path} --command=BuildModel --pdb-dir={pdb_path} --pdb={pdb_name} --mutant-file={mutation_path} --output-dir={output_dir}"
    # foldx_command = f"{foldx_path} --command=Stability --pdb-dir={pdb_path} --pdb={pdb_name}  --output-dir={output_dir}"
    print(foldx_command)
    subprocess.run(foldx_command, shell=True)

def extract_energy_values(file_content):
    # Split the content by lines
    lines = file_content.split('\n')
    
    # Initialize a list to store the energy values
    energy_values = []
    
    # Regular expression to match lines that start with the PDB identifier and capture the total energy value
    energy_pattern = re.compile(r'^\w+\s+\d+\s+([-+]?\d*\.\d+|\d+)')
    
    for line in lines:
        match = energy_pattern.match(line)
        if match:
            # Extract the total energy value and convert it to float
            total_energy = float(match.group(1))
            energy_values.append(total_energy)
    
    return energy_values

def create_summery(fraction):
    foldx_ds_path = base_path + "/data/Processed_K50_dG_datasets/foldx"
    pdb_path = base_path + "/data/Processed_K50_dG_datasets/AlphaFold_model_PDBs"
    foldx_ds = os.listdir(foldx_ds_path)
    
    # Calculate start and end indices for this quarter
    total_len = len(foldx_ds)
    quarter_size = total_len // 10
    start_idx = (fraction - 1) * quarter_size
    end_idx = start_idx + quarter_size if fraction < 10 else total_len
    
    foldx_ds = foldx_ds[start_idx:end_idx]
    summery_df = pd.DataFrame()
    
    for protein in tqdm(foldx_ds):
        foldx_enerrgy_path = f'{foldx_ds_path}/{protein}/foldx_output2/Average_{protein}.fxout'
        # check if the file exists
        if not os.path.exists(foldx_enerrgy_path):
            continue
        with open(foldx_enerrgy_path, 'r') as f:
            file_content = f.read()
        df = pd.read_csv(f'{foldx_ds_path}/{protein}/foldx.csv')
        # Add FoldX energy values to the dataframe
        df['foldx_dg'] = extract_energy_values(file_content)
        df.to_csv(f'{foldx_ds_path}/{protein}/foldx.csv', index=False)
        summery_df = pd.concat([summery_df, df])
    summery_df.to_csv(f'{foldx_ds_path}/foldx_summery_{fraction}.csv', index=False)

if __name__ == "__main__":
    # test_foldx()
    fraction = int(sys.argv[1])
    print(fraction)
    validate_foldx(fraction)
    create_summery(fraction)