import ref


def check(workdir):
    from ggufsize.calc import model_total_bytes

    max_rel_err = 0.0
    for cfg in ref.CONFIGS:
        tensors = cfg["tensors"]
        ftype = cfg.get("tensors", [{}])[0].get("ftype", 0)
        for leave in [True, False]:
            want = ref.model_total_bytes(tensors, ftype, leave_output=leave)
            got = model_total_bytes(tensors, ftype, leave_output=leave)
            if want == 0:
                err = 0.0 if got == 0 else 1.0
            else:
                err = abs(got - want) / abs(want)
            if err > max_rel_err:
                max_rel_err = err

    return {"rel_err": float(max_rel_err)}
