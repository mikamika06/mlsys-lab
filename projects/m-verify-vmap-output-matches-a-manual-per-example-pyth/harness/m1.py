import ref


def check(workdir):
    from batchspmd.vmap import per_example_loop, verify_vmap_matches

    out = {"max_abs_err": 0.0, "correct_match_count": 0.0}
    max_err_found = 0.0
    correct_count = 0

    for i, (fn_s, fn_b) in enumerate(ref.TEST_FUNCS):
        x = ref.TEST_BATCHES[i % len(ref.TEST_BATCHES)]

        want_loop = ref.per_example_loop(fn_s, x)
        got_loop = per_example_loop(fn_s, x)
        loop_err = float(ref.np.max(ref.np.abs(want_loop - got_loop)))

        want_err = ref.verify_vmap_matches(fn_s, fn_b, x)
        got_err = verify_vmap_matches(fn_s, fn_b, x)

        err_diff = abs(want_err - got_err)
        cur_err = max(loop_err, err_diff)
        if cur_err > max_err_found:
            max_err_found = cur_err

        if cur_err <= 1e-5:
            correct_count += 1
        elif "_note" not in out:
            out["_note"] = f"fn {i}: loop error {loop_err:.6e}, match err diff {err_diff:.6e}"

    out["max_abs_err"] = float(max_err_found)
    out["correct_match_count"] = float(correct_count)
    return out
