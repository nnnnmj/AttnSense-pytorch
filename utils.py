#### input_pipeline ####
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd

def read_csv(filenames, W, dim):
    # Assuming 'filenames' contains the path to the CSV file
    # and 'W', 'dim', and 'OUT_DIM' are appropriately defined

    # Read the CSV file using pandas
    df = pd.read_csv(filenames, header=None)

    # Extract features and labels from the dataframe
    features = torch.tensor(df.iloc[:, :W * dim].values, dtype=torch.float32)
    features = features.view(-1, W, dim)
    labels = torch.tensor(df.iloc[:, W * dim:].values, dtype=torch.float32)

    return features, labels

class CustomDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        example, label = read_csv(self.filenames[idx])
        return example, label

def input_pipeline(filenames, batch_size, shuffle_sample=True, num_epochs=None):
    dataset = CustomDataset(filenames)
    if shuffle_sample:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

#### output_pipeline ####