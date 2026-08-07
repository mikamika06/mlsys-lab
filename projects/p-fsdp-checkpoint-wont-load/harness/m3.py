import os
import torch
import ref

def check(workdir):
    m = {"converted_ok": 0.0}
    tmp_dir = os.path.join(workdir, "tmp_m3_ckpt")
    ref.generate_dummy_checkpoint(tmp_dir)
    out_path = os.path.join(tmp_dir, "combined.pt")

    try:
        import fsdp_ckpt.converter as conv
        conv.convert_to_portable(tmp_dir, out_path)
        if os.path.exists(out_path):
            data = torch.load(out_path)
            if "model.weight" in data and data["model.weight"].shape[0] == 8:
                m["converted_ok"] = 1.0
    except Exception:
        pass

    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)
    return m
