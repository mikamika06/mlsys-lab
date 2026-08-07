import ref

def check(workdir):
    from serving.simulator import simulate_eviction

    out = {
        "lru_rel_err": 0.0,
        "lfu_rel_err": 0.0,
        "lus_rel_err": 0.0
    }

    for reqs in ref.WORKLOADS:
        cap = 15
        w_lru = ref.simulate_eviction(reqs, cap, "lru")
        w_lfu = ref.simulate_eviction(reqs, cap, "lfu")
        w_lus = ref.simulate_eviction(reqs, cap, "lus")

        g_lru = simulate_eviction(reqs, cap, "lru")
        g_lfu = simulate_eviction(reqs, cap, "lfu")
        g_lus = simulate_eviction(reqs, cap, "lus")

        out["lru_rel_err"] = max(out["lru_rel_err"], abs(g_lru - w_lru) / (w_lru + 1e-9))
        out["lfu_rel_err"] = max(out["lfu_rel_err"], abs(g_lfu - w_lfu) / (w_lfu + 1e-9))
        out["lus_rel_err"] = max(out["lus_rel_err"], abs(g_lus - w_lus) / (w_lus + 1e-9))

    return out
