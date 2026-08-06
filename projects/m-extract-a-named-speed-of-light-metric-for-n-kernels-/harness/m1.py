import ref

def check(workdir):
    from solparser.parser import extract_sol_metrics
    csv_data = ref.generate_sample_csv()
    kernels = ["kernel_matmul", "kernel_softmax", "kernel_layer norm"]
    metric = "sm__throughput.avg.pct_of_peak_sustained_elapsed"

    try:
        got = extract_sol_metrics(csv_data, kernels, metric)
    except Exception as e:
        return {"metrics_matched": 0.0, "_note": f"Exception raised: {e}"}

    expected = {
        "kernel_matmul": 78.5,
        "kernel_softmax": 62.1,
        "kernel_layer norm": 45.0
    }

    matched = 0
    for k in kernels:
        if got.get(k) == expected.get(k):
            matched += 1

    out = {"metrics_matched": float(matched)}
    if matched < len(kernels):
        out["_note"] = f"Expected {expected}, got {got}"
    return out
