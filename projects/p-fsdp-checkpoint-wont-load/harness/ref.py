import os
import torch

def generate_dummy_checkpoint(tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    t = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    torch.save({"model.weight": t[:4]}, os.path.join(tmp_dir, "rank_0.pt"))
    torch.save({"model.weight": t[4:]}, os.path.join(tmp_dir, "rank_1.pt"))
    with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
        f.write('{"world_size": 2}')
    return t
