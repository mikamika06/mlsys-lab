import numpy as np

def convert_tensor(shards_data, split_dim):
    return np.concatenate(shards_data, axis=split_dim)
