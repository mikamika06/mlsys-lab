import ref


def check(workdir):
    from shm.bytes import compute_shm_bytes

    cases = ref.get_m1_cases()
    max_rel_err = 0.0
    for case in cases:
        got = compute_shm_bytes(
            case["block_m"],
            case["block_n"],
            case["block_k"],
            case["num_stages"],
            case["dtype"],
        )
        want = case["expected_bytes"]
        err = abs(got - want) / max(1.0, float(want))
        if err > max_rel_err:
            max_rel_err = err
    return {"rel_err": float(max_rel_err)}
