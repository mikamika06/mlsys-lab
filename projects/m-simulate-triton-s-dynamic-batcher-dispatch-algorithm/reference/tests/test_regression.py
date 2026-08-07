import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import triton_batcher.optimize as opt

def test_optimizer_respects_throughput_floor():
    orig_sim = opt.simulate
    orig_met = opt.measure_metrics

    def fake_simulate(arr, mx, pref, delay, comp):
        return [{"pref": pref, "delay": delay}]

    def fake_metrics(arr, disps, comp):
        cfg = disps[0]
        if cfg["delay"] == 1000:
            return {"throughput_req_sec": 10.0, "p99_queue_delay_us": 500.0}
        return {"throughput_req_sec": 100.0, "p99_queue_delay_us": 5000.0}

    opt.simulate = fake_simulate
    opt.measure_metrics = fake_metrics

    try:
        res = opt.optimize_config([0], 8, [[4]], [1000, 5000], 50.0, lambda x: x)
        assert res is not None, "Optimizer returned None"
        assert res["delay_us"] == 5000, f"Expected config with delay 5000, got {res}"
    finally:
        opt.simulate = orig_sim
        opt.measure_metrics = orig_met
