import sys
import os
import torch

sys.path.insert(0, ".")
from fsdp_ckpt.converter import convert_to_portable, restore_from_portable
from fsdp_ckpt.loader import verify_loss

def test_checkpoint_roundtrip():
    tmp_dir = "tmp_test_ckpt"
    os.makedirs(tmp_dir, exist_ok=True)

    dummy_tensor = torch.randn(8, 4)
    shard1 = {"weight": dummy_tensor[:4]}
    shard2 = {"weight": dummy_tensor[4:]}

    torch.save(shard1, os.path.join(tmp_dir, "rank_0.pt"))
    torch.save(shard2, os.path.join(tmp_dir, "rank_1.pt"))

    out_path = os.path.join(tmp_dir, "portable.pt")
    convert_to_portable(tmp_dir, out_path)

    r0 = restore_from_portable(out_path, target_world_size=4, rank=0)
    assert r0["weight"].shape[0] == 2

    full_state = {"weight": dummy_tensor}
    restored_full = torch.cat([
        restore_from_portable(out_path, target_world_size=2, rank=0)["weight"],
        restore_from_portable(out_path, target_world_size=2, rank=1)["weight"]
    ], dim=0)

    assert verify_loss(full_state, {"weight": restored_full})

    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)
