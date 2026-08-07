import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, tokenizer, num_samples=20):
        self.input_ids = torch.randint(0, 100, (num_samples, 16))
        self.labels = torch.randint(0, 100, (num_samples, 16))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx]
        }


def get_reference_setup():
    return DummyDataset(None)
