import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from triton_remapping.cache_sim import simulate_block_loads
        from triton_remapping.search import find_optimal_group_size
    except Exception as e:
        return {"simulation_matched": 0.0, "search_matched": 0.0, "_note": f"Import error: {e}"}

    sim_matched = True
    for cfg in ref.TEST_CONFIGS:
        m, n, g, cap = cfg["num_pid_m"], cfg["num_pid_n"], cfg["group_size_m"], cfg["cap"]
        want_loads = ref.simulate_block_loads(m, n, g, cap)
        got_loads = simulate_block_loads(m, n, g, cap)
        if want_loads != got_loads:
            sim_matched = False
            break

    search_matched = True
    for cfg in ref.TEST_CONFIGS:
        m, n, cap = cfg["num_pid_m"], cfg["num_pid_n"], cfg["cap"]
        want_best = ref.find_optimal_group_size(m, n, cap, max_group_size=16)
        got_best = find_optimal_group_size(m, n, cap, max_group_size=16)
        if want_best != got_best:
            search_matched = False
            break

    return {
        "simulation_matched": 1.0 if sim_matched else 0.0,
        "search_matched": 1.0 if search_matched else 0.0
    }
