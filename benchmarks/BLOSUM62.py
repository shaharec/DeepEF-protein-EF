import blosum as bl
import pandas as pd
import numpy as np


def test_blosum():
    matrix = bl.BLOSUM(62)
    val = matrix["A"]["Y"]
    assert val == -2
    print("BLOSUM62 test passed")   

def blusom62_mutation(csv_path,result_path):
    """save change in mutation uaing blosum62 matrix"""
    matrix = bl.BLOSUM(62)
    df = pd.read_csv(csv_path)
    # remove inster and delete
    # Remove rows where 'mut_type' contains 'ins' or 'del'
    df = df[~df["mut_type"].str.contains("ins|del|wt")]
    # Remove rows with more than one mutation (split by ':')
    df = df[df["mut_type"].str.split(':').apply(len) == 1]
    # Get blosu62 values for each mutation
    df['blosum62'] = df['mut_type'].apply(lambda x: matrix[x[0]][x[-1]])
    # Save the result
    df.to_csv(result_path, index=False)
    print("BLOSUM62 mutation values saved: ", result_path)
    

if __name__ == "__main__":
    blusom62_path = "./data/Processed_K50_dG_datasets/blusom62.csv"
    result_path = "./data/Processed_K50_dG_datasets/blusom62_values.csv"
    blusom62_mutation(blusom62_path,result_path)