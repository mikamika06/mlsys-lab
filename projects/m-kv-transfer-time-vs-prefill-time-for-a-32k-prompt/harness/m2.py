import ref


def check(workdir):
    from disagg.sizing import allocate_pd_nodes, compute_pd_ratio
    from disagg.sim import simulate_pipeline

    out = {"ratio_rel_err": 1.0, "sim_rel_err": 1.0}
    max_ratio_err = 0.0

    for p_ms, t_ms, d_ms, g_tok in ref.SIZING_CONFIGS:
        ref_r = ref.compute_pd_ratio(p_ms, t_ms, d_ms, g_tok)
        got_r = compute_pd_ratio(p_ms, t_ms, d_ms, g_tok)
        err = abs(got_r - ref_r) / max(abs(ref_r), 1e-9)
        if err > max_ratio_err:
            max_ratio_err = err

    for tot_nodes, p_tot, d_tot in ref.ALLOC_CONFIGS:
        ref_alloc = ref.allocate_pd_nodes(tot_nodes, p_tot, d_tot)
        got_alloc = allocate_pd_nodes(tot_nodes, p_tot, d_tot)
        if got_alloc != ref_alloc:
            out["_note"] = f"allocate_pd_nodes mismatch: got {got_alloc}, want {ref_alloc}"
            return out

    max_sim_err = 0.0
    for reqs, np, nd, m_cfg, h_cfg in ref.SIM_CONFIGS:
        ref_s = ref.simulate_pipeline(reqs, np, nd, m_cfg, h_cfg)
        got_s = simulate_pipeline(reqs, np, nd, m_cfg, h_cfg)
        for k in ("avg_ttft_ms", "avg_tpot_ms", "total_makespan_ms", "p_utilization", "d_utilization"):
            if k not in got_s:
                out["_note"] = f"missing key {k} in simulate_pipeline result"
                return out
            err_s = abs(got_s[k] - ref_s[k]) / max(abs(ref_s[k]), 1e-9)
            if err_s > max_sim_err:
                max_sim_err = err_s

    out["ratio_rel_err"] = max_ratio_err
    out["sim_rel_err"] = max_sim_err
    return out
