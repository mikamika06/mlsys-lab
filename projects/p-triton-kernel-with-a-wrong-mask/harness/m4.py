import sys


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"sweep_completed": 0.0, "all_sizes_correct": 0.0}
    try:
        import triton_mask.kernel as k

        sweep = k.run_boundary_sweep(60, 130, BLOCK_SIZE=64)
        expected_keys = set(range(60, 131))
        if set(sweep.keys()) == expected_keys:
            res["sweep_completed"] = 1.0
            if all(sweep.values()):
                res["all_sizes_correct"] = 1.0
    except Exception:
        pass
    return res
