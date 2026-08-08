import os
import numpy as np
import ref

def check(workdir):
    m = {"restored_different_ranks": 0.0}
    tmp_dir = os.path.join(workdir, "tmp_m4_ckpt")
    ref.generate_dummy_checkpoint(tmp_dir)
    out_path = os.path.join(tmp_dir, "combined.npy")

    try:
        import fsdp_ckpt.converter as conv
        conv.convert_to_portable(tmp_dir, out_path)
        r0 = conv.restore_from_portable(out_path, 4, 0)
        if "model.weight" in r0 and r0["model.weight"].shape[0] == 2:
            m["restored_different_ranks"] = 1.0
    except Exception:
        pass

    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)
    return m
