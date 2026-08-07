import triton.optimize as opt

def test_respects_throughput_floor():
    orig_sim = opt.simulate
    orig_met = opt.calculate_metrics

    try:
        def fake_sim(arr, max_b, pref, delay, comp):
            return [{"delay_used": delay}]

        def fake_met(arr, batches, comp):
            d = batches[0]["delay_used"]
            if d == 10:
                return {"throughput": 100, "p99_queue_delay": 10}
            if d == 20:
                return {"throughput": 200, "p99_queue_delay": 20}
            return {"throughput": 0, "p99_queue_delay": 999}

        opt.simulate = fake_sim
        opt.calculate_metrics = fake_met

        best = opt.optimize_delay([], 8, [4], [10, 20], 150.0, lambda b: 1)
        assert best == 20, f"Expected 20 to meet throughput floor, got {best}"
    finally:
        opt.simulate = orig_sim
        opt.calculate_metrics = orig_met
