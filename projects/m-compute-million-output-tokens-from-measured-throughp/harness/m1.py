import sys
import os

def check(workdir):
    sys.path.insert(0, os.path.join(workdir, "harness"))
    sys.path.insert(0, workdir)
    import ref

    out = {"cost_rel_err": 1.0}
    try:
        from capacity.cost import compute_cost_per_million_tokens
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    cost_cases, _, _ = ref.generate_test_cases()
    max_err = 0.0

    for idx, case in enumerate(cost_cases):
        want = ref.ref_compute_cost_per_million_tokens(case["throughput"], case["price"])
        try:
            got = compute_cost_per_million_tokens(case["throughput"], case["price"])
        except Exception as e:
            out["_note"] = f"Execution error on case {idx}: {e}"
            return out

        err = abs(got - want) / want
        if err > max_err:
            max_err = err

    out["cost_rel_err"] = float(max_err)
    return out
