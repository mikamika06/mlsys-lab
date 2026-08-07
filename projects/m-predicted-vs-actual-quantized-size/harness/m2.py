import ref


def check(workdir):
    from quant.modes import analyze_op_compatibility, compare_int8_vs_int16x8

    out = {"modes_matched": 0.0, "ops_matched": 0.0}
    modes_ok = True
    ops_ok = True

    for model in ref.MODELS:
        for mode in ("dynamic_range", "int8", "int16x8"):
            want_analysis = ref.analyze_op_compatibility(model, mode)
            got_analysis = analyze_op_compatibility(model, mode)
            if got_analysis != want_analysis:
                ops_ok = False

        want_comp = ref.compare_int8_vs_int16x8(model)
        got_comp = compare_int8_vs_int16x8(model)
        if got_comp != want_comp:
            modes_ok = False

    out["modes_matched"] = 1.0 if modes_ok else 0.0
    out["ops_matched"] = 1.0 if ops_ok else 0.0
    return out
