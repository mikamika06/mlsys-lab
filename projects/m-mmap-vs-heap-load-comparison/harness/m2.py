import ref


def check(workdir):
    from weightloader.attribution import attribute_regression
    from weightloader.dedup import compute_dedup_savings
    from reference.weightloader.attribution import attribute_regression as ref_attr
    from reference.weightloader.dedup import compute_dedup_savings as ref_dedup

    out = {"attribution_match": 0.0, "dedup_match": 0.0}
    cfg = ref.CONFIGS[0]

    b = {"peak_rss": 10000, "page_faults": 10}
    n = {"peak_rss": 15000, "page_faults": 15}

    got_attr = attribute_regression(b, n)
    want_attr = ref_attr(b, n)
    if got_attr == want_attr:
        out["attribution_match"] = 1.0

    got_dedup = compute_dedup_savings(cfg)
    want_dedup = ref_dedup(cfg)
    if got_dedup == want_dedup:
        out["dedup_match"] = 1.0

    return out
