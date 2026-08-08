import os
import numpy as np

def generate_dummy_checkpoint(tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    t = np.arange(16, dtype=np.float32).reshape(8, 2)
    np.save(os.path.join(tmp_dir, "rank_0.npy"), {"model.weight": t[:4]})
    np.save(os.path.join(tmp_dir, "rank_1.npy"), {"model.weight": t[4:]})
    with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
        f.write('{"world_size": 2}')
    return t
