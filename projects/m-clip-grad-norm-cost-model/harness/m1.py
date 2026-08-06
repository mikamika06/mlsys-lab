import ref


def check(workdir):
    from gradclip.cost import estimate_cost
    out = {"cost_matched": 0.0, "configs": float(len(ref.get_test_cases()))}
    ok = 0
    for i, params in enumerate(ref.get_test_cases()):
        want = ref.estimate_cost(params, 1.0)
        got = estimate_cost(params, 1.0)
        if isinstance(got, dict) and got.get("total_elements") == want["total_elements"] and got.get("num_tensors") == want["num_tensors"]:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, reference {want}"
    out["cost_matched"] = float(ok)
    return out
