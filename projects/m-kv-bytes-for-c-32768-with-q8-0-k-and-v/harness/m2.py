import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from kvquant.memory import calculate_kv_cache_bytes, evaluate_perplexity_delta

    out = {"bytes_matched": 0.0, "ppl_matched": 0.0}

    test_configs = [
        (32, 8, 128, 32768, "f16"),
        (32, 8, 128, 32768, "q8_0"),
        (32, 8, 128, 32768, "q4_0"),
        (80, 8, 128, 32768, "q8_0"),
        (24, 16, 128, 16384, "q4_0"),
    ]

    bytes_ok = True
    for cfg in test_configs:
        want = ref.reference_calculate_kv_cache_bytes(*cfg)
        got = calculate_kv_cache_bytes(*cfg)
        if want != got:
            bytes_ok = False
            out["_note"] = f"Bytes mismatch for {cfg}: want {want}, got {got}"
            break

    if bytes_ok:
        out["bytes_matched"] = 1.0

    ppl_configs = [
        (6.5, "f16", 32768),
        (6.5, "q8_0", 32768),
        (6.5, "q4_0", 32768),
        (5.2, "q8_0", 65536),
    ]

    ppl_ok = True
    for cfg in ppl_configs:
        want = ref.reference_evaluate_perplexity_delta(*cfg)
        got = evaluate_perplexity_delta(*cfg)
        if abs(want - got) > 1e-4:
            ppl_ok = False
            out["_note"] = f"Perplexity mismatch for {cfg}: want {want}, got {got}"
            break

    if ppl_ok:
        out["ppl_matched"] = 1.0

    return out
