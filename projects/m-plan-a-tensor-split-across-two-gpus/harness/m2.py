import ref


def check(workdir):
    from tensorsplit.split import compute_tensor_split

    out = {"rel_err": 1.0}
    errs = []
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_tensor_split(cfg)
        got = compute_tensor_split(cfg)
        if not isinstance(got, list) or len(got) != 2:
            return {"rel_err": 1.0, "_note": f"config {i}: invalid return format {got}"}
        diff = abs(got[0] - want[0]) + abs(got[1] - want[1])
        errs.append(diff)

    max_err = max(errs) if errs else 1.0
    out["rel_err"] = float(max_err)
    return out
