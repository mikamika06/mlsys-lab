import os
import ref

def check(workdir):
    m = {"structure_parsed": 0.0}
    tmp_dir = os.path.join(workdir, "tmp_m1_ckpt")
    ref.generate_dummy_checkpoint(tmp_dir)

    try:
        import fsdp_ckpt.converter as conv
        res = conv.parse_structure(tmp_dir)
        if isinstance(res, dict) and "shards" in res and len(res["shards"]) == 2:
            m["structure_parsed"] = 1.0
    except Exception:
        pass

    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)
    return m
