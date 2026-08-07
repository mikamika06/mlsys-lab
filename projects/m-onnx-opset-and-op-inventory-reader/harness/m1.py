import ref


def check(workdir):
    from onnx_reader.reader import scan_opsets

    out = {"opsets_matched": 0.0, "configs": float(len(ref.MODELS))}
    ok = 0

    for i, model in enumerate(ref.MODELS):
        want = ref.scan_opsets(model)
        try:
            got = scan_opsets(model)
            if want == got:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"model {i}: got {got}, reference {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"model {i} failed: {type(e).__name__}: {str(e)}"

    out["opsets_matched"] = float(ok)
    return out
