import ref


def check(workdir):
    from diag.roofline import analyze_roofline

    out = {"roofline_matched": 0.0}
    ok = True
    for i, mix in enumerate(ref.OP_MIXES):
        want = ref.analyze_roofline(mix, ref.HARDWARE_SPECS)
        got = analyze_roofline(mix, ref.HARDWARE_SPECS)
        if not isinstance(got, dict):
            ok = False
            out["_note"] = f"mix {i}: expected dict return type"
            break
        for k in ("intensity", "knee_point", "attained_tflops", "bound"):
            if k not in got:
                ok = False
                out["_note"] = f"mix {i}: missing key {k}"
                break
            if k == "bound":
                if got[k] != want[k]:
                    ok = False
                    out["_note"] = f"mix {i}: bound mismatch {got[k]} != {want[k]}"
                    break
            elif abs(float(got[k]) - float(want[k])) > 1e-4:
                ok = False
                out["_note"] = f"mix {i}: field {k} mismatch {got[k]} != {want[k]}"
                break
        if not ok:
            break
    if ok:
        out["roofline_matched"] = 1.0
    return out
