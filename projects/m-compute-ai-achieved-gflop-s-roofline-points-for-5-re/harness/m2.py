import ref


def check(workdir):
    from roofline.calc import compute_roofline_points, classify_kernels, compare_attention

    kernels, peak_flops, peak_bw = ref.get_test_data()
    metrics = compute_roofline_points(kernels, peak_flops, peak_bw)

    std_m = next(m for m in metrics if m["name"] == "standard_attention")
    flash_m = next(m for m in metrics if m["name"] == "flash_attention")

    try:
        comp = compare_attention(std_m, flash_m)
        classes = classify_kernels(metrics)
    except Exception as e:
        return {"classification_match": 0.0, "_note": f"raised exception: {e}"}

    match = 1.0 if comp == "flash" and len(classes) == 5 else 0.0
    return {"classification_match": float(match)}
