import ref


def check(workdir):
    from onnx_reader.reader import estimate_ort_savings, scan_ops

    out = {"ops_matched": 0.0, "savings_matched": 0.0}
    ok_ops = 0
    ok_sav = 0

    for i, model in enumerate(ref.MODELS):
        want_ops = ref.scan_ops(model)
        want_sav = ref.estimate_ort_savings(model)

        try:
            got_ops = scan_ops(model)
            if want_ops == got_ops:
                ok_ops += 1
            elif "_note_ops" not in out:
                out["_note_ops"] = f"model {i}: ops got {got_ops}, ref {want_ops}"
        except Exception as e:
            if "_note_ops" not in out:
                out["_note_ops"] = f"model {i} ops failed: {e}"

        try:
            got_sav = estimate_ort_savings(model)
            if want_sav == got_sav:
                ok_sav += 1
            elif "_note_sav" not in out:
                out["_note_sav"] = f"model {i}: savings got {got_sav}, ref {want_sav}"
        except Exception as e:
            if "_note_sav" not in out:
                out["_note_sav"] = f"model {i} savings failed: {e}"

    out["ops_matched"] = float(ok_ops)
    out["savings_matched"] = float(ok_sav)
    return out
