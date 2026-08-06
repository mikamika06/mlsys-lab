import ref


def check(workdir):
    from moeoffload.budget import calculate_vram, max_context_length

    out = {"budgets_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    total_test_cases = 0

    for cfg in ref.CONFIGS:
        for offload in [True, False]:
            total_test_cases += 1
            want_vram = ref.calculate_vram(cfg, offload)
            got_vram = calculate_vram(cfg, offload)

            vram_total = want_vram + 2 * 1024 * 1024 * 1024
            want_ctx = ref.max_context_length(vram_total, cfg, offload)
            got_ctx = max_context_length(vram_total, cfg, offload)

            if want_vram == got_vram and want_ctx == got_ctx:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"mismatch for config offload={offload}: got vram={got_vram}, ctx={got_ctx}, want vram={want_vram}, ctx={want_ctx}"

    out["budgets_matched"] = float(ok)
    return out
