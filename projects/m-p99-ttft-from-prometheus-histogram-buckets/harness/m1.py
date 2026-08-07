import math
import ref


def check(workdir):
    from kvobs.histogram import calculate_histogram_quantile

    out = {"quantiles_matched": 0.0}
    total = len(ref.HISTOGRAM_TEST_CASES)
    matched = 0

    for case in ref.HISTOGRAM_TEST_CASES:
        q = case["q"]
        buckets = case["buckets"]
        want = ref.calculate_histogram_quantile(q, buckets)
        try:
            got = calculate_histogram_quantile(q, buckets)
            if math.isclose(got, want, rel_tol=1e-5, abs_tol=1e-5):
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"q={q}: got {got}, expected {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"error on q={q}: {e}"

    if matched == total:
        out["quantiles_matched"] = 1.0
    return out
