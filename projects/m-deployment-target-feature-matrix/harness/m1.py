import ref

def check(workdir):
    from exporttools.matrix import validate_target_matrix
    out = {"matrix_matched": 0.0}
    tests = [
        ("iOS15", ["add", "relu"], True),
        ("iOS15", ["gelu"], False),
        ("iOS17", ["scaled_dot_product_attention"], True),
        ("iOS16", ["flash_attention"], False)
    ]
    ok = True
    for target, ops, want_valid in tests:
        res = validate_target_matrix(target, ops)
        if res.get("valid") != want_valid:
            ok = False
            break
    if ok:
        out["matrix_matched"] = 1.0
    return out
