import ref


def check(workdir):
    from moeplan.roofline import classify_regime, compute_crossover_batch

    out = {"roofline_matched": 0.0}
    matched = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want_cross = ref.ref_crossover(cfg)
        want_regime = ref.ref_classify(cfg)

        try:
            got_cross = compute_crossover_batch(
                cfg["num_ranks"], cfg["hidden_dim"], cfg["ffn_inter_dim"],
                cfg["bus_gbps"], cfg["compute_tflops"]
            )
            got_regime = classify_regime(
                cfg["tokens"], cfg["num_ranks"], cfg["hidden_dim"],
                cfg["ffn_inter_dim"], cfg["bus_gbps"], cfg["compute_tflops"]
            )
            if abs(got_cross - want_cross) / max(1.0, want_cross) < 1e-4 and got_regime == want_regime:
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"cfg {i}: cross got {got_cross}, want {want_cross}; regime got {got_regime}, want {want_regime}"
        except Exception as e:  # noqa: BLE001
            if "_note" not in out:
                out["_note"] = f"cfg {i} raised {type(e).__name__}: {e}"

    out["roofline_matched"] = float(matched)
    return out
