import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from roofline.classify import classify_kernel, max_achievable_gflops

    out = {"classifications_matched": 0.0, "performance_matched": 0.0}

    cls_ok = True
    perf_ok = True

    for test in ref.CLASSIFY_TESTS:
        dev = ref.DEVICES[test["device_idx"]]
        ridge = dev["peak_gflops"] / dev["bandwidth_gbps"]
        want_cls = ref.classify_kernel(test["ai"], ridge)
        got_cls = classify_kernel(test["ai"], ridge)

        if got_cls != want_cls:
            cls_ok = False
            out["_note"] = f"classify_kernel({test['ai']}, {ridge:.2f}) got {got_cls}, want {want_cls}"
            break

        want_perf = ref.max_achievable_gflops(test["ai"], dev["peak_gflops"], dev["bandwidth_gbps"])
        got_perf = max_achievable_gflops(test["ai"], dev["peak_gflops"], dev["bandwidth_gbps"])

        if abs(got_perf - want_perf) > 1e-4:
            perf_ok = False
            out["_note"] = f"max_achievable_gflops got {got_perf}, want {want_perf}"
            break

    out["classifications_matched"] = 1.0 if cls_ok else 0.0
    out["performance_matched"] = 1.0 if perf_ok else 0.0

    return out
