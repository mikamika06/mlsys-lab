import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from scalezero.simulator import simulate_scale_to_zero

    out = {"cases_matched": 0.0, "total_cases": len(ref.TRAFFIC_PATTERNS) * 12}
    matched = 0

    for traffic in ref.TRAFFIC_PATTERNS:
        for timeout in [1, 2, 5, 10]:
            for latency in [1, 2, 3]:
                want_exp, want_cold = ref.simulate_scale_to_zero(traffic, timeout, latency)
                try:
                    got_exp, got_cold = simulate_scale_to_zero(traffic, timeout, latency)
                except Exception as e:
                    out["_note"] = f"Error: {e}"
                    sys.path.pop(0)
                    return out

                if got_exp == want_exp and got_cold == want_cold:
                    matched += 1
                elif "_note" not in out:
                    out["_note"] = f"Failed on timeout={timeout}, latency={latency}. got ({got_exp}, {got_cold}), want ({want_exp}, {want_cold})"

    out["cases_matched"] = float(matched)
    sys.path.pop(0)
    return out
