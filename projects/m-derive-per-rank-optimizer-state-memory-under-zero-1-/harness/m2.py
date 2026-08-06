import ref


def check(workdir):
    from zeromem.stages import total_memory
    out = {"stages_matched": 0.0, "total": 0.0}
    ok = 0
    total = 0
    for p in ref.PARAMS_LIST:
        for w in ref.WORLD_SIZES:
            for stage in (1, 2, 3):
                for act in ref.ACTIVATIONS:
                    total += 1
                    param_mem = 2 * p
                    grad_mem = 2 * p
                    opt_mem = 12 * p
                    if stage == 1:
                        want = param_mem + grad_mem + (opt_mem // w) + act
                    elif stage == 2:
                        want = param_mem + (grad_mem // w) + (opt_mem // w) + act
                    elif stage == 3:
                        want = (param_mem // w) + (grad_mem // w) + (opt_mem // w) + act
                    got = total_memory(p, w, stage, act)
                    if got == want:
                        ok += 1
    out["stages_matched"] = float(ok)
    out["total"] = float(total)
    return out
