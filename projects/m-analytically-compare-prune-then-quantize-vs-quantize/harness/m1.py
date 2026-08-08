import ref


def check(workdir):
    from pqutils.analysis import analytical_error_comparison
    fixtures = ref.generate_fixtures()
    ok = 0
    out = {"bound_matches": 0.0, "fixtures": float(len(fixtures))}
    for idx, fx in enumerate(fixtures):
        try:
            res = analytical_error_comparison(fx["weights"], fx["sparsity"], fx["q_bits"])
            if isinstance(res, dict) and "prune_then_quantize_error" in res and "quantize_then_prune_error" in res:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"fixture {idx} returned invalid structure: {res}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"fixture {idx} raised {type(e).__name__}: {str(e)[:100]}"
    out["bound_matches"] = float(ok)
    return out
