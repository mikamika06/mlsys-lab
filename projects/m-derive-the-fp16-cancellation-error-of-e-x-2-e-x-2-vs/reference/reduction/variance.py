import numpy as np

def one_pass_variance(x):
    x = x.astype(np.float16)
    x_sq = (x * x).astype(np.float16)
    mean_x = np.mean(x, dtype=np.float16)
    mean_x_sq = np.mean(x_sq, dtype=np.float16)
    var = mean_x_sq - (mean_x * mean_x).astype(np.float16)
    return var.astype(np.float16)

def two_pass_variance(x):
    x = x.astype(np.float16)
    mean_x = np.mean(x, dtype=np.float16)
    diff = (x - mean_x).astype(np.float16)
    sq = (diff * diff).astype(np.float16)
    return np.mean(sq, dtype=np.float16)
