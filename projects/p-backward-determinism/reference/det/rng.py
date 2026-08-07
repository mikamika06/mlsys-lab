import numpy as np

def fix_state(seed):
    np.random.seed(seed)
    return np.random.get_state()[1]
