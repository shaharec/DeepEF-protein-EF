import pandas as pd

ROSETTA_DATASET_PATH = "./data/Processed_K50_dG_datasets/rosetta.csv"
VALIDATION_DATASET_PATH = "./data/Processed_K50_dG_datasets/validation_protein.csv"

class benckmark_datasets:
    """Benckmark datasets for the project.
    - Foldx model
    - Rosetta model"""
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.only_validation = False
        self.validation_dataset = pd.read_csv(VALIDATION_DATASET_PATH)
        self.dataset = self.load_dataset()
        
    def load_dataset(self):
        if self.dataset_name == "Foldx":
            return self.load_foldx_dataset()
        elif self.dataset_name == "Rosetta":
            return self.load_rosetta_dataset()
        else:
            raise ValueError("Invalid dataset name")
        
    def get_item(self, index):
        return self.dataset.iloc[index].to_dict()
    
    def load_rosetta_dataset(self):
        df = pd.read_csv(ROSETTA_DATASET_PATH)
        df['pdb_name'] = df['name'].apply(lambda x: x.split(".")[0])
        # save only whats in the validation set
        if self.only_validation:
            df = df[df['pdb_name'].isin(self.validation_dataset['name'])]
        return df
        
def test_benckmark_datasets():
    dataset = benckmark_datasets("Rosetta")
    print(dataset.get_item(0))
    print("length of dataset: ", len(dataset.dataset))
    
if __name__ == "__main__":
    test_benckmark_datasets()