import ref

def check(workdir):
    from dtypecheck.roundtrip import quantify_roundtrip_loss
    tensors = ref.generate_extreme_tensors()
    ok = 1
    for t in tensors:
        want = ref.quantify_roundtrip_loss(t)
        got = quantify_roundtrip_loss(t)
        if abs(want - got) > 1e-5:
            ok = 0
    return {"loss_match": float(ok)}
