import ref


def check(workdir):
    from vllmsched.metrics import calculate_throughput, measure_concurrency_sweep

    out = {"throughput_curve_valid": 0.0}

    levels = [1, 4, 8, 16]
    user_res = measure_concurrency_sweep(
        ref.gen_test_requests,
        levels,
        num_blocks=16,
        block_size=16,
        max_tokens=64,
    )
    ref_res = ref.measure_concurrency_sweep(
        ref.gen_test_requests,
        levels,
        num_blocks=16,
        block_size=16,
        max_tokens=64,
    )

    match = True
    for conc in levels:
        if abs(user_res.get(conc, 0.0) - ref_res[conc]) > 1e-4:
            match = False
            out["_note"] = (
                f"At concurrency {conc}, expected throughput {ref_res[conc]}, "
                f"got {user_res.get(conc)}"
            )
            break

    if match:
        out["throughput_curve_valid"] = 1.0

    return out
