import ref


def check(workdir):
    from batchspmd.benchmark import benchmark_vmap_speedup

    out = {
        "speedup_measured": 0.0,
        "ratio_accuracy": 0.0,
        "num_batches_measured": 0.0,
    }

    class MockTimer:
        def __init__(self):
            self.curr = 0.0

        def __call__(self):
            self.curr += 0.01
            return self.curr

    mock_timer = MockTimer()
    fn_s, fn_b = ref.TEST_FUNCS[0]
    res = benchmark_vmap_speedup(fn_s, fn_b, ref.TEST_BATCHES, timer=mock_timer)

    if not isinstance(res, dict):
        out["_note"] = f"expected dict, got {type(res).__name__}"
        return out

    out["num_batches_measured"] = float(len(res))
    if len(res) == 0:
        out["_note"] = "returned empty dict"
        return out

    accurate_ratios = 0
    all_keys_valid = True
    for b_size, metrics in res.items():
        if not isinstance(metrics, dict) or "speedup" not in metrics:
            all_keys_valid = False
            break
        t_loop = metrics.get("loop_time", 0.0)
        t_vmap = metrics.get("vmap_time", 0.0)
        sp = metrics.get("speedup", 0.0)
        expected_sp = t_loop / t_vmap if t_vmap > 0 else 1.0
        if abs(sp - expected_sp) < 1e-5:
            accurate_ratios += 1

    if all_keys_valid:
        out["speedup_measured"] = 1.0
    if accurate_ratios == len(res):
        out["ratio_accuracy"] = 1.0

    return out
