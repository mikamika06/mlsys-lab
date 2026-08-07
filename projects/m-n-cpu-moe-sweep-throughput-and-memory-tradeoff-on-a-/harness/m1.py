import ref


def check(workdir):
    from moe_sweep.sweep import simulate_sweep
    cfg = ref.get_model_config()
    n_vals = ref.get_n_cpu_values()
    want = ref.simulate_sweep(cfg, n_vals)
    got = simulate_sweep(cfg, n_vals)
    match = 1.0 if got == want else 0.0
    return {"sweep_match": match}
