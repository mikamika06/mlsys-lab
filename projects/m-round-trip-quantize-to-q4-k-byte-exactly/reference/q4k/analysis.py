import numpy as np


def dominant_subblock(superblock):
    flat = np.asarray(superblock, dtype=np.float32).flatten()
    sub_size = 32
    num_subs = len(flat) // sub_size
    mse_list = []
    for s in range(num_subs):
        sub = flat[s * sub_size:(s + 1) * sub_size]
        mean_val = np.mean(sub)
        mse = float(np.mean((sub - mean_val) ** 2))
        mse_list.append(mse)
    return int(np.argmax(mse_list))
