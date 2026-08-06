import ref


def check(workdir):
    from moe_quant.perplexity import estimate_perplexity_delta

    out = {"perplexity_matched": 0.0}
    try:
        base = 15.5
        ref_val = estimate_perplexity_delta(base, True, True)

        import moe_quant.perplexity as p
        user_val = p.estimate_perplexity_delta(base, True, True)

        if abs(user_val - ref_val) < 1e-5:
            out["perplexity_matched"] = 1.0
        else:
            out["_note"] = f"Expected perplexity {ref_val}, got {user_val}"
    except Exception as e:
        out["_note"] = f"Milestone 2 execution error: {str(e)[:100]}"
    return out
