import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from numprec.properties import get_format_properties, compute_relative_error

    out = {"properties_matched": 0.0}
    matched = 0

    for dt in ref.TEST_DTYPES:
        want = ref.get_ref_properties(dt)
        try:
            got = get_format_properties(dt)
        except Exception as e:
            out["_note"] = f"Failed get_format_properties({dt}): {e}"
            return out

        keys = ["exponent_bits", "mantissa_bits", "max_val", "min_pos_normal", "min_pos_subnormal", "eps"]
        all_ok = True
        for k in keys:
            v_want = want[k]
            v_got = got.get(k, None)
            if v_got is None or abs(v_got - v_want) / (abs(v_want) + 1e-15) > 1e-3:
                all_ok = False
                out["_note"] = f"Mismatch in {dt} for key {k}: got {v_got}, want {v_want}"
                break

        if all_ok:
            rel_err = compute_relative_error(1.0 + 2.0**(-11), dt)
            if dt == "fp16" and abs(rel_err) < 1e-10:
                all_ok = False
                out["_note"] = f"Relative error calculation for {dt} failed to reflect precision loss"

        if all_ok:
            matched += 1

    out["properties_matched"] = float(matched)
    return out
