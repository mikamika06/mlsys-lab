import os
import numpy as np
import ref

def check(workdir):
    m = {"loss_matches": 0.0}
    tmp_dir = os.path.join(workdir, "tmp_m5_ckpt")
    orig_t = ref.generate_dummy_checkpoint(tmp_dir)
    out_path = os.path.join(tmp_dir, "combined.npy")

    try:
        import fsdp_ckpt.converter as conv
        import fsdp_ckpt.loader as loader
        conv.convert_to_portable(tmp_dir, out_path)

        restored_chunks = []
        target_ws = 4
        for r in range(target_ws):
            restored_chunks.append(conv.restore_from_portable(out_path, target_ws, r)["model.weight"])
        full_restored = np.concatenate(restored_chunks, axis=0)

        orig_state = {"model.weight": orig_t}
        rest_state = {"model.weight": full_restored}

        if loader.verify_loss(orig_state, rest_state):
            m["loss_matches"] = 1.0
    except Exception:
        pass

    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)
    return m
