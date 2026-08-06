import ref

def check(workdir):
    from mixplan.tensor_check import verify_1d_tensors
    out = {"tensors_f32": 0.0, "size_ratio": 1.0}
    ftypes = ["Q4_0", "Q8_0", "F32"]
    tensors_good = [
        {"name": "bias", "shape": [32], "ftype_map": {"Q4_0": "F32", "Q8_0": "F32", "F32": "F32"}},
        {"name": "weight", "shape": [32, 32], "ftype_map": {"Q4_0": "Q4_0", "Q8_0": "Q8_0", "F32": "F32"}}
    ]
    tensors_bad = [
        {"name": "bias", "shape": [32], "ftype_map": {"Q4_0": "Q4_0", "Q8_0": "F32", "F32": "F32"}},
        {"name": "weight", "shape": [32, 32], "ftype_map": {"Q4_0": "Q4_0", "Q8_0": "Q8_0", "F32": "F32"}}
    ]
    res_good = verify_1d_tensors(tensors_good, ftypes)
    res_bad = verify_1d_tensors(tensors_bad, ftypes)
    if res_good and not res_bad:
        out["tensors_f32"] = 1.0
        out["size_ratio"] = 1.5
    return out
