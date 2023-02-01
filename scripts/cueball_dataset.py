import numpy as np
import torch
from torch.utils.data import Dataset


class CueballDataset(Dataset):
    """Dataset for Cueball tasks."""

    def __init__(self, npz_file: str, transform=None) -> None:
        self.npz_file = npz_file
        self.dataset = np.load(npz_file)
        self.transform = transform

    def __len__(self):
        return self.dataset[list(self.dataset.keys())[0]].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {k: v[idx] for k, v in self.dataset.items()}
