import ref


def check(workdir):
    from attention.traffic import compute_hbm_bytes

    out = {"traffic_matched": 0.0}
    ok = True
    for cfg in ref.CONFIGS:
        want = ref.ref_hbm_bytes(
            cfg["batch_size"], cfg["seq_len"], cfg["num_heads"], cfg["head_dim"], cfg["dtype_bytes"]
        )
        try:
            got = compute_hbm_bytes(
                cfg["batch_size"], cfg["seq_len"], cfg["num_heads"], cfg["head_dim"], cfg["dtype_bytes"]
            )
        except Exception as e:
            out["_note"] = f"raised exception: {e}"
            return out
        if abs(got - want) > 1e-5:
            ok = False
            out["_note"] = f"got {got}, want {want}"
            break
    if ok:
        out["traffic_matched"] = 1.0
    return out
