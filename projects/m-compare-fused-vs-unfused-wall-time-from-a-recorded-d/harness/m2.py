import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from bench_analysis.report import generate_benchmark_summary

    records = ref.generate_test_records()
    matched = 1.0
    min_throughput_ratio = 100.0

    for idx, rec in enumerate(records):
        try:
            got = generate_benchmark_summary(rec)
            expected = ref.reference_summary(rec)

            for key in [
                "unfused_mean_ms",
                "fused_mean_ms",
                "speedup",
                "time_saved_ms",
                "unfused_gbps",
                "fused_gbps",
            ]:
                if abs(got[key] - expected[key]) > 1e-3:
                    matched = 0.0
                    return {
                        "metrics_matched": 0.0,
                        "throughput_ratio": 0.0,
                        "_note": f"Key {key} mismatch on record {idx}: got {got[key]}, expected {expected[key]}",
                    }

            ratio = got.get("throughput_ratio", 0.0)
            if ratio < min_throughput_ratio:
                min_throughput_ratio = ratio

        except Exception as e:
            return {
                "metrics_matched": 0.0,
                "throughput_ratio": 0.0,
                "_note": f"Execution error: {e}",
            }

    return {
        "metrics_matched": matched,
        "throughput_ratio": min_throughput_ratio,
    }
