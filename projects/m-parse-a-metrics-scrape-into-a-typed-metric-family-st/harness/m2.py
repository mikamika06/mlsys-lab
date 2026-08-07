import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from vllm_metrics.stats import compute_p99_ttft
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"Failed to import compute_p99_ttft: {e}"}

    max_err = 0.0
    for i, text in enumerate(ref.TEST_SCRAPES):
        fam = ref.parse_scrape(text).get("vllm:time_to_first_token_seconds")
        want = ref.compute_p99_ttft(fam)
        try:
            got = compute_p99_ttft(fam)
        except Exception as e:
            return {
                "rel_err": 1.0,
                "_note": f"compute_p99_ttft raised on scrape {i}: {e}",
            }

        if set(got.keys()) != set(want.keys()):
            return {
                "rel_err": 1.0,
                "_note": f"Scrape {i}: label keys mismatch in quantile output",
            }

        for k in want:
            got_p99, got_err = got[k]
            want_p99, want_err = want[k]
            e1 = abs(got_p99 - want_p99) / (1.0 + abs(want_p99))
            e2 = abs(got_err - want_err) / (1.0 + abs(want_err))
            err = max(e1, e2)
            if err > max_err:
                max_err = err

    return {"rel_err": max_err}
