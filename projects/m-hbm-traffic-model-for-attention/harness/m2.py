import ref


def check(workdir):
    from attention.intensity import compute_arithmetic_intensity

    out = {"intensity_matched": 0.0}
    ok = True
    for cfg in ref.CONFIGS:
        want = ref.ref_arithmetic_intensity(
            cfg["batch_size"], cfg["seq_len"], cfg["num_heads"], cfg["head_dim"], cfg["dtype_bytes"]
        )
        try:
            got = compute_arithmetic_intensity(
                cfg["batch_size"], cfg["seq_len"], cfg["num_heads"], cfg["head_dim"], cfg["dtype_bytes"]
            )
        except Exception as e:
            out["_note"] = f"raised exception: {e}"
            return out
        if abs(got - want) / (abs(want) + 1e-9) > 1e-4:
            ok = False
            out["_note"] = f"got {got}, want {want}"
            break
    if ok:
        out["intensity_matched"] = 1.0
    return out
