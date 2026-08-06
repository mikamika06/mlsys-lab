import ref

def check(workdir):
    from profiling.derivation import derive_ai_formula
    cases = ref.get_test_cases()
    max_rel_err = 0.0

    for case in cases:
        want = ref.derive_formula(case)
        got = derive_ai_formula(
            case["params"],
            case["hidden_size"],
            case["num_layers"],
            case["total_weight_bytes"],
            case["prompt_tokens"]
        )
        for k in want:
            w_val = want[k]
            g_val = got.get(k, 0.0)
            err = abs(g_val - w_val) / (abs(w_val) + 1e-9)
            if err > max_rel_err:
                max_rel_err = err

    out = {"rel_err_derivation": float(max_rel_err)}
    if max_rel_err > 0.01:
        out["_note"] = f"Derivation relative error {max_rel_err:.4f} exceeds 0.01 threshold."
    return out
