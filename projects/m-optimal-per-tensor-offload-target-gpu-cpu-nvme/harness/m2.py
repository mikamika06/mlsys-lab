import ref


def check(workdir):
    from offload_target.adam_bench import compare_adam_performance
    from offload_target.log_parser import analyze_nvme_logs

    out = {"adam_benchmarks_matched": 0.0, "log_metrics_matched": 0.0}

    tensor_sizes = [1048576, 16777216, 134217728]
    thread_counts = [1, 4, 8, 16]

    want_adam = ref.compare_adam_performance(tensor_sizes, thread_counts)
    got_adam = compare_adam_performance(tensor_sizes, thread_counts)

    if got_adam == want_adam:
        out["adam_benchmarks_matched"] = 1.0
    else:
        out["_note_adam"] = f"Adam mismatch: got {got_adam[:1]}, want {want_adam[:1]}"

    want_logs = ref.analyze_nvme_logs(ref.LOG_SAMPLES)
    got_logs = analyze_nvme_logs(ref.LOG_SAMPLES)

    if got_logs == want_logs:
        out["log_metrics_matched"] = 1.0
    else:
        out["_note_logs"] = f"Log parser mismatch: got {got_logs}, want {want_logs}"

    return out
