import ref


def check(workdir):
    from vllmbench.arrival import generate_poisson_arrivals

    rate = 20.0
    duration = 10.0
    seed = 42

    want = ref.compute_reference_arrivals(rate, duration, seed=seed)
    try:
        got = generate_poisson_arrivals(rate, duration, seed=seed)
    except Exception as e:
        return {"arrival_rate_matched": 0.0, "_note": f"raised exception: {e}"}

    if not isinstance(got, list) or len(got) == 0:
        return {"arrival_rate_matched": 0.0, "_note": "returned empty or non-list"}

    got_arr = ref.np.array(got)
    want_arr = ref.np.array(want)

    if len(got_arr) != len(want_arr):
        diff_len_ratio = min(len(got_arr), len(want_arr)) / max(len(got_arr), len(want_arr))
        return {"arrival_rate_matched": float(diff_len_ratio >= 0.95)}

    max_diff = ref.np.max(ref.np.abs(got_arr - want_arr)) if len(got_arr) > 0 else 0.0
    score = 1.0 if max_diff < 1e-5 else 0.0
    return {"arrival_rate_matched": score}
