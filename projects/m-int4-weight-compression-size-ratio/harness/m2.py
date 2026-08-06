import ref


def check(workdir):
    from compression.footprint import memory_footprint
    from compression.awq import perplexity_delta

    out = {"footprint_matched": 0.0, "perplexity_delta_matched": 0.0}
    cfg = ref.CONFIGS[0]

    want_fp = ref.compute_footprint(
        cfg["weights_count"], cfg["bits"], cfg["group_size"], cfg["scale_bits"]
    )
    try:
        got_fp = memory_footprint(
            cfg["weights_count"], cfg["bits"], cfg["group_size"], cfg["scale_bits"]
        )
    except Exception as e:
        out["_note"] = f"footprint raised {type(e).__name__}"
        return out

    if abs(got_fp - want_fp) < 1e-3:
        out["footprint_matched"] = 1.0
    else:
        out["_note"] = f"footprint mismatch: got {got_fp}, want {want_fp}"
        return out

    df_ppl = 12.5
    awq_ppl = 8.2
    want_delta = ref.compute_perplexity_delta(df_ppl, awq_ppl)
    try:
        got_delta = perplexity_delta(df_ppl, awq_ppl)
    except Exception as e:
        out["_note"] = f"perplexity_delta raised {type(e).__name__}"
        return out

    if abs(got_delta - want_delta) < 1e-5:
        out["perplexity_delta_matched"] = 1.0
    else:
        out["_note"] = f"delta mismatch: got {got_delta}, want {want_delta}"

    return out
