import sys


def check(workdir):
    sys.path.insert(0, workdir)
    m = {"offload_time_correct": 0.0}

    try:
        from zero_planner.estimator import ZeroEstimator

        est = ZeroEstimator(num_params=10**8, bytes_per_param=2, bytes_per_optim_state=12)
        base_t = 0.5
        ws = 4
        pcie_bw = 16.0

        t_no_offload = est.step_latency_with_offload(base_t, stage=2, world_size=ws, pcie_bandwidth_gbps=pcie_bw, cpu_offload=False)
        t_offload = est.step_latency_with_offload(base_t, stage=2, world_size=ws, pcie_bandwidth_gbps=pcie_bw, cpu_offload=True)

        offloaded_bytes = (10**8 * 12) / ws
        expected_transfer_time = (2.0 * offloaded_bytes) / (pcie_bw * 1e9)
        expected_t = base_t + expected_transfer_time

        if abs(t_no_offload - base_t) < 1e-6 and abs(t_offload - expected_t) < 1e-5:
            m["offload_time_correct"] = 1.0
    except Exception:
        pass

    return m
