import ref

def check(workdir):
    from onednn_diag.isa_sweep import analyze_k_sweep
    from onednn_diag.profiler import compute_primitive_dominance
    from reference.onednn_diag.isa_sweep import analyze_k_sweep as ref_sweep
    from reference.onednn_diag.profiler import compute_primitive_dominance as ref_prof

    sweep_data = ref.generate_k_sweep_data(seed=202)
    got_sweep = analyze_k_sweep(sweep_data)
    want_sweep = ref_sweep(sweep_data)

    sweep_ok = 1.0 if got_sweep == want_sweep else 0.0

    logs = ref.generate_verbose_logs(seed=303)
    got_prof = compute_primitive_dominance(logs)
    want_prof = ref_prof(logs)

    prof_ok = 1.0 if got_prof == want_prof else 0.0

    return {
        "isa_sweeps_matched": sweep_ok,
        "profiler_breakdown_matched": prof_ok
    }
