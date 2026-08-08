import sys
import os
import numpy as np

sys.path.insert(0, ".")
from fsdp_ckpt.converter import convert_to_portable, restore_from_portable
from fsdp_ckpt.loader import verify_loss

def test_checkpoint_roundtrip():
    tmp_dir = "tmp_test_ckpt"
    os.makedirs(tmp_dir, exist_ok=True)

    dummy_array = np.arange(32, dtype=np.float32).reshape(8, 4)
    shard1 = {"weight": dummy_array[:4]}
    shard2 = {"weight": dummy_array[4:]}

    np.save(os.path.join(tmp_dir, "rank_0.npy"), shard1)
    np.save(os.path.join(tmp_dir, "rank_1.npy"), shard2)

    out_path = os.path.join(tmp_dir, "portable.npy")
    convert_to_portable(tmp_dir, out_path)

    r0 = restore_from_portable(out_path, target_world_size=4, rank=0)
    assert r0["weight"].shape[0] == 2

    full_state = {"weight": dummy_array}
    restored_full = np.concatenate([
        restore_from_portable(out_path, target_world_size=2, rank=0)["weight"],
        restore_from_portable(out_path, target_world_size=2, rank=1)["weight"]
    ], axis=0)

    assert verify_loss(full_state, {"weight": restored_full})

    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)
