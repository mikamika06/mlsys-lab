import ref

def check(workdir):
    from modelfile.validate import validate_modelfile
    out = {"validations_matched": 0.0, "total": 3.0}
    ok = 0

    v1, l1, _ = validate_modelfile(ref.VALID_MF)
    if v1 is True:
        ok += 1

    v2, l2, _ = validate_modelfile(ref.INVALID_MF)
    if v2 is False and l2 == 2:
        ok += 1

    v3, l3, _ = validate_modelfile(ref.INVALID_PARAM_MF)
    if v3 is False and l3 == 2:
        ok += 1

    out["validations_matched"] = float(ok)
    return out
