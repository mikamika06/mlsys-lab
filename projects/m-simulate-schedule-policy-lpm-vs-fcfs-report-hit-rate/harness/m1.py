import ref

def check(workdir):
    from serving.simulator import simulate_schedule

    out = {
        "lpm_hit_rel_err": 0.0,
        "fcfs_hit_rel_err": 0.0,
        "lpm_wait_err": 0.0,
        "fcfs_wait_err": 0.0
    }

    for reqs in ref.WORKLOADS:
        w_lpm_hit, w_lpm_wait = ref.simulate_schedule(reqs, "lpm")
        w_fcfs_hit, w_fcfs_wait = ref.simulate_schedule(reqs, "fcfs")

        g_lpm_hit, g_lpm_wait = simulate_schedule(reqs, "lpm")
        g_fcfs_hit, g_fcfs_wait = simulate_schedule(reqs, "fcfs")

        lpm_err = abs(g_lpm_hit - w_lpm_hit) / (w_lpm_hit + 1e-9)
        fcfs_err = abs(g_fcfs_hit - w_fcfs_hit) / (w_fcfs_hit + 1e-9)

        out["lpm_hit_rel_err"] = max(out["lpm_hit_rel_err"], lpm_err)
        out["fcfs_hit_rel_err"] = max(out["fcfs_hit_rel_err"], fcfs_err)
        out["lpm_wait_err"] = max(out["lpm_wait_err"], float(abs(g_lpm_wait - w_lpm_wait)))
        out["fcfs_wait_err"] = max(out["fcfs_wait_err"], float(abs(g_fcfs_wait - w_fcfs_wait)))

    return out
