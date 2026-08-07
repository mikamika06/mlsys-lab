import sys


def check(workdir):
    sys.path.insert(0, workdir)
    m = {"comm_volume_correct": 0.0}

    try:
        from zero_planner.estimator import ZeroEstimator
        import ref

        est = ZeroEstimator(num_params=10**8, bytes_per_param=2, bytes_per_optim_state=12)
        oracle = ref.get_oracle_estimates(num_params=10**8, world_size=4, act_mem=0)

        c1 = est.comm_bytes_per_step(stage=1, world_size=4)
        c2 = est.comm_bytes_per_step(stage=2, world_size=4)
        c3 = est.comm_bytes_per_step(stage=3, world_size=4)

        if (abs(c1 - oracle["comm1_2"]) < 1e-4 and
            abs(c2 - oracle["comm1_2"]) < 1e-4 and
            abs(c3 - oracle["comm3"]) < 1e-4):
            m["comm_volume_correct"] = 1.0
    except Exception:
        pass

    return m
