import ref


def check(workdir):
    from nsysprof.warmup import compute_warmup_metrics

    kernels = ref.generate_synthetic_kernels()
    exp_idle, exp_util = ref.compute_warmup_ref(kernels)
    got_idle, got_util = compute_warmup_metrics(kernels)

    err_idle = abs(got_idle - exp_idle) / (abs(exp_idle) if exp_idle != 0 else 1.0)
    err_util = abs(got_util - exp_util) / (abs(exp_util) if exp_util != 0 else 1.0)
    rel_err = max(err_idle, err_util)

    return {"rel_err": float(rel_err)}
