import ref


def check(workdir):
    from optbudget import bytes as opt_bytes

    out = {"bytes_matched": 0.0}
    test_types = ["adam_fp32", "adam_8bit", "sgd"]
    ok = True
    for t in test_types:
        want = ref.derive_optimizer_bytes_per_param(t)
        try:
            got = opt_bytes.derive_optimizer_bytes_per_param(t)
        except Exception as e:
            ok = False
            out["_note"] = f"optimizer {t} raised error: {type(e).__name__}"
            break
        if abs(float(got) - float(want)) > 1e-5:
            ok = False
            out["_note"] = f"optimizer {t}: got {got}, reference {want}"
            break

    out["bytes_matched"] = 1.0 if ok else 0.0
    return out
