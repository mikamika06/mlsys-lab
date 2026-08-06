import ref

def check(workdir):
    from qlora_mem.nf4 import compare_nf4_uniform
    t = ref.generate_tensor()
    want = ref.compare_nf4_uniform(t)
    got = compare_nf4_uniform(t)
    out = {"nf4_verified": 0.0}
    if got.get("nf4_beats_uniform") == want.get("nf4_beats_uniform") and abs(got.get("nf4_mse", 0) - want.get("nf4_mse", 0)) < 1e-5:
        out["nf4_verified"] = 1.0
    else:
        out["_note"] = f"got {got}, reference {want}"
    return out
