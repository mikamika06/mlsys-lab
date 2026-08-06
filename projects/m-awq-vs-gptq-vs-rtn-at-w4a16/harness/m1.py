import ref


def check(workdir):
    from awqsim.compare import compare_methods

    fixtures = ref.generate_fixtures()
    out = {"evals_matched": 0.0}
    ok = 0

    tests = [
        (fixtures["W1"], fixtures["X1"]),
        (fixtures["W2_a"], fixtures["X2"]),
        (fixtures["W2_b"], fixtures["X2"]),
    ]

    for W, X in tests:
        want = ref.compare_methods(W, X)
        got = compare_methods(W, X)
        if got is not None and isinstance(got, dict):
            keys_ok = all(k in got for k in ("rtn_mse", "gptq_mse", "awq_mse"))
            if keys_ok:
                close_rtn = abs(got["rtn_mse"] - want["rtn_mse"]) < 1e-4
                close_gptq = abs(got["gptq_mse"] - want["gptq_mse"]) < 1e-4
                close_awq = abs(got["awq_mse"] - want["awq_mse"]) < 1e-4
                if close_rtn and close_gptq and close_awq:
                    ok += 1
                elif "_note" not in out:
                    out["_note"] = f"mismatch: got {got}, reference {want}"
        elif "_note" not in out:
            out["_note"] = f"invalid return shape or type: {got}"

    out["evals_matched"] = float(ok)
    return out
