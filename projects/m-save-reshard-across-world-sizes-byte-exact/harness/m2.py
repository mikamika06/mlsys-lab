import ref


def check(workdir):
    from dcp.memory import estimate_memory_and_time

    configs = ref.get_test_configs()
    memory_ok = 1
    time_ok = 1
    for cfg in configs:
        tb = cfg["total_bytes"]
        ws = cfg["world_sizes"][0]
        bw = cfg["bandwidth"]

        want_mem, want_time = tb // ws, tb / (bw * 1024 * 1024 * 1024)
        got_mem, got_time = estimate_memory_and_time(tb, ws, bw)

        if int(got_mem) != int(want_mem):
            memory_ok = 0
        if abs(float(got_time) - float(want_time)) > 1e-5:
            time_ok = 0

    return {"memory_match": float(memory_ok), "time_match": float(time_ok)}
