import ref


def check(workdir):
    from shapes.verifier import ConstraintViolationError, Dim, verify_range

    out = {"in_range_match": 0.0, "out_range_match": 0.0, "fix_msg_match": 0.0}

    ok = 0
    for d, sz in ref.M1_IN_RANGE:
        dim = Dim(d["name"], d["min"], d["max"])
        try:
            if verify_range(dim, sz) is True:
                ok += 1
        except NotImplementedError:
            return out
        except Exception:
            pass
    if ok == len(ref.M1_IN_RANGE):
        out["in_range_match"] = 1.0

    ok_err = 0
    ok_msg = 0
    for d, sz in ref.M1_OUT_RANGE:
        dim = Dim(d["name"], d["min"], d["max"])
        try:
            verify_range(dim, sz)
        except ConstraintViolationError as e:
            ok_err += 1
            want_msg = ref.get_msg(d["name"], sz, d["min"], d["max"])
            if e.suggested_fix() == want_msg:
                ok_msg += 1
        except NotImplementedError:
            return out
        except Exception:
            pass

    if ok_err == len(ref.M1_OUT_RANGE):
        out["out_range_match"] = 1.0
    if ok_msg == len(ref.M1_OUT_RANGE):
        out["fix_msg_match"] = 1.0

    return out
