import sys


def check(workdir):
    sys.path.insert(0, workdir)
    m = {"zero1_mem_correct": 0.0, "zero2_mem_correct": 0.0, "zero3_mem_correct": 0.0}

    try:
        from zero_planner.estimator import ZeroEstimator
        import ref

        est = ZeroEstimator(num_params=10**8, bytes_per_param=2, bytes_per_optim_state=12)
        oracle = ref.get_oracle_estimates(num_params=10**8, world_size=8, act_mem=500 * (1024**2))

        z1 = est.memory_zero1(world_size=8, act_mem_per_gpu=500 * (1024**2))
        z2 = est.memory_zero2(world_size=8, act_mem_per_gpu=500 * (1024**2))
        z3 = est.memory_zero3(world_size=8, act_mem_per_gpu=500 * (1024**2))

        if abs(z1 - oracle["z1"]) < 1e-4:
            m["zero1_mem_correct"] = 1.0
        if abs(z2 - oracle["z2"]) < 1e-4:
            m["zero2_mem_correct"] = 1.0
        if abs(z3 - oracle["z3"]) < 1e-4:
            m["zero3_mem_correct"] = 1.0
    except Exception:
        pass

    return m
