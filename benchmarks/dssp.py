import pandas as pd
from Bio.PDB import DSSP, PDBParser

def dssp_to_dataframe(pdb_file, dssp_file):
    # Parse PDB file
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file)

    # Use DSSP to parse the DSSP file
    model = structure[0]
    dssp = DSSP(model, dssp_file)

    # Extract DSSP information
    data = []
    for key, values in dssp.property_dict.items():
        chain, res_num, res_code = key
        aa, sec_str, phi, psi, asa, hbonds = values[0], values[1], values[4], values[5], values[3], values[6:]
        
        data.append({
            "Chain": chain,
            "Residue Number": res_num,
            "Residue Code": res_code,
            "Amino Acid": aa,
            "Secondary Structure": sec_str,
            "Phi": phi,
            "Psi": psi,
            "ASA": asa,
            "HBonds": hbonds
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

pdb_file = "path/to/your.pdb"
dssp_file = "path/to/your.dssp"

df = dssp_to_dataframe(pdb_file, dssp_file)
print(df.head())