import ref


def check(workdir):
    from gguf_pred.inventory import predict_output_size

    out = {"inventories_matched": 0.0}
    ok = 0
    for i, inv in enumerate(ref.INVENTORIES):
        want = ref.predict_output_size(inv)
        got = predict_output_size(inv)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"inventory {i}: got {got}, reference {want}"
    out["inventories_matched"] = float(ok)
    return out
