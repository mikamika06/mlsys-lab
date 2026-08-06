import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from bandwidth.roofline import (
            analyze_kernel_execution,
            compute_arithmetic_intensity,
            compute_roofline_bound,
        )
    except Exception as e:
        return {"rel_err": 1.0, "classification_acc": 0.0, "_note": f"Failed to import roofline functions: {e}"}

    max_err = 0.0
    class_correct = 0
    total_class = 0

    for case in ref.EXECUTION_CASES:
        flops = case["flops"]
        bytes_tr = case["bytes_transferred"]
        exec_time = case["execution_time_sec"]
        peak_tflops = case["peak_tflops"]
        peak_gbps = case["peak_gbps"]

        want_intensity = ref.compute_arithmetic_intensity(flops, bytes_tr)
        try:
            got_intensity = compute_arithmetic_intensity(flops, bytes_tr)
        except Exception as e:
            return {"rel_err": 1.0, "classification_acc": 0.0, "_note": f"compute_arithmetic_intensity raised {e}"}

        err = abs(got_intensity - want_intensity) / max(abs(want_intensity), 1e-9)
        max_err = max(max_err, err)

        want_bound = ref.compute_roofline_bound(want_intensity, peak_tflops, peak_gbps)
        got_bound = compute_roofline_bound(got_intensity, peak_tflops, peak_gbps)

        for key in ("attainable_tflops", "knee_intensity"):
            w = want_bound[key]
            g = got_bound.get(key, 0.0)
            err = abs(g - w) / max(abs(w), 1e-9)
            max_err = max(max_err, err)

        total_class += 1
        if got_bound.get("is_memory_bound") == want_bound["is_memory_bound"]:
            class_correct += 1

        want_analysis = ref.analyze_kernel_execution(flops, bytes_tr, exec_time, peak_tflops, peak_gbps)
        got_analysis = analyze_kernel_execution(flops, bytes_tr, exec_time, peak_tflops, peak_gbps)

        for key, w in want_analysis.items():
            if isinstance(w, bool):
                total_class += 1
                if got_analysis.get(key) == w:
                    class_correct += 1
            else:
                g = got_analysis.get(key, 0.0)
                err = abs(g - w) / max(abs(w), 1e-9)
                max_err = max(max_err, err)

    acc = float(class_correct / total_class) if total_class > 0 else 0.0
    return {"rel_err": float(max_err), "classification_acc": acc}
