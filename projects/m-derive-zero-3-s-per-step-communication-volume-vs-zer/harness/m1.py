import ref


def check(workdir):
    from zeroutil.comm import compute_zero_communication_volume

    out = {"comm_derived": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        p = cfg["params_bytes"]
        w = cfg["world_size"]
        got = compute_zero_communication_volume(p, w)
        n = float(w)
        want_z1 = 2.0 * p * (n - 1.0) / n
        want_z3 = 2.0 * p * (n - 1.0) / n + 2.0 * p
        if (
            isinstance(got, dict)
            and abs(got.get("zero1_comm_bytes", -1) - want_z1) < 1e-5
            and abs(got.get("zero3_comm_bytes", -1) - want_z3) < 1e-5
        ):
            ok += 1
    out["comm_derived"] = float(ok)
    return out
