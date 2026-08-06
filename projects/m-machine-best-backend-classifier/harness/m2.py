import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    sys.path.insert(0, workdir)
    from fa_backend.cost import measure_fallback_cost

    out = {"costs_matched": 0.0}

    matched = 0
    total = float(len(ref.MACHINES) * len(ref.INPUT_SPECS))

    for m in ref.MACHINES:
        for spec in ref.INPUT_SPECS:
            target_b = ref.ref_classify(m, spec)
            fallback_b = "MATH_FALLBACK"

            want_cost = ref.ref_measure_cost(target_b, fallback_b, m, spec)
            got_cost = measure_fallback_cost(target_b, fallback_b, m, spec)

            lat_diff = abs(got_cost.get("latency_penalty_ratio", 0.0) - want_cost["latency_penalty_ratio"])
            mem_diff = abs(got_cost.get("memory_overhead_bytes", 0) - want_cost["memory_overhead_bytes"])

            if lat_diff < 1e-4 and mem_diff == 0:
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"cost mismatch on {m['id']}/{spec['id']}: got {got_cost}, want {want_cost}"

    out["costs_matched"] = float(matched == total)
    return out
