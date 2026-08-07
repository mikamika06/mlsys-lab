import math
import ref


def check(workdir):
    from hardening.cost import calculate_worst_case_cost

    out = {"cost_estimates_matched": 0.0, "total_configs": float(len(ref.REQUEST_CONFIGS))}
    matched = 0

    for i, cfg in enumerate(ref.REQUEST_CONFIGS):
        expected = ref.oracle_calculate_worst_case_cost(cfg, ref.PROFILE_PARAMS)
        try:
            got = calculate_worst_case_cost(cfg, ref.PROFILE_PARAMS)
            if math.isclose(got, expected, rel_tol=1e-5, abs_tol=1e-5):
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"Config {i}: expected {expected}, got {got}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Config {i} raised {type(e).__name__}: {e}"

    out["cost_estimates_matched"] = float(matched)
    return out
