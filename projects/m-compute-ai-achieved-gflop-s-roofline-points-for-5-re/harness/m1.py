import ref


def check(workdir):
    from roofline.calc import compute_roofline_points

    kernels, peak_flops, peak_bw = ref.get_test_data()
    want = compute_roofline_points(kernels, peak_flops, peak_bw)

    try:
        got = compute_roofline_points(kernels, peak_flops, peak_bw)
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"raised exception: {e}"}

    if not isinstance(got, list) or len(got) != len(want):
        return {"rel_err": 1.0, "_note": "invalid return structure"}

    max_err = 0.0
    for w_item, g_item in zip(want, got):
        for k in ["ai", "gflops", "efficiency"]:
            w_val = w_item[k]
            g_val = g_item.get(k, 0.0)
            err = abs(w_val - g_val) / (abs(w_val) + 1e-9)
            if err > max_err:
                max_err = err

    return {"rel_err": float(max_err)}
