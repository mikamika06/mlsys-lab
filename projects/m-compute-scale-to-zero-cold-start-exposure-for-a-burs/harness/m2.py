import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from scalezero.optimizer import find_optimal_timeout

    out = {"cases_matched": 0.0, "total_cases": len(ref.TRAFFIC_PATTERNS) * 6}
    matched = 0

    for traffic in ref.TRAFFIC_PATTERNS:
        for latency in [1, 2]:
            for ratio in [0.05, 0.1, 0.5]:
                want = ref.find_optimal_timeout(traffic, latency, ratio)
                try:
                    got = find_optimal_timeout(traffic, latency, ratio)
                except Exception as e:
                    out["_note"] = f"Error: {e}"
                    sys.path.pop(0)
                    return out

                if got == want:
                    matched += 1
                elif "_note" not in out:
                    out["_note"] = f"Failed on latency={latency}, ratio={ratio}. got {got}, want {want}"

    out["cases_matched"] = float(matched)
    sys.path.pop(0)
    return out
