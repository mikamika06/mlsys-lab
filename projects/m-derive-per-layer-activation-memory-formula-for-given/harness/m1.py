import ref

def check(workdir):
    from actmem.formula import compute_layer_activation_bytes
    matched = 1
    for cfg in ref.CONFIGS:
        want = ref.compute_layer_activation_bytes(cfg["b"], cfg["s"], cfg["h"], cfg["heads"], cfg["dtype_bytes"])
        got = compute_layer_activation_bytes(cfg["b"], cfg["s"], cfg["h"], cfg["heads"], cfg["dtype_bytes"])
        if abs(got - want) > 1e-5:
            matched = 0
            break
    return {"formula_matched": float(matched)}
