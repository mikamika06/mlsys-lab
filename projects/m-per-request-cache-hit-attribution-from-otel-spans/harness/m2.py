import ref


def check(workdir):
    from otel_cache.attribution import extract_attribution
    from otel_cache.metrics import compute_efficiency

    all_spans = []
    for cfg in ref.CONFIGS:
        all_spans.extend(cfg["spans"])

    attr = extract_attribution(all_spans)
    got = compute_efficiency(attr)
    want = ref.compute_efficiency(ref.extract_attribution(all_spans))

    if want == 0.0:
        rel = 0.0 if got == 0.0 else 1.0
    else:
        rel = abs(got - want) / abs(want)

    return {"rel_error": float(rel)}
