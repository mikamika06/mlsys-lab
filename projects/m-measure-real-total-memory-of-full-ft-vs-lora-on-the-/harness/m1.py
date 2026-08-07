import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from ftmem.memory import estimate_memory_footprint
    except Exception as e:
        return {
            "full_ft_match": 0.0,
            "lora_bf16_match": 0.0,
            "size_ratio": 1.0,
            "_note": f"Import error: {e}",
        }

    full_ok = 0
    lora_ok = 0
    max_ratio = 0.0
    total = len(ref.CONFIGS)

    for i, cfg in enumerate(ref.CONFIGS):
        lora_cfg = ref.LORA_CONFIGS[i % len(ref.LORA_CONFIGS)]

        want_full = ref.estimate_memory_footprint(
            cfg, mode="full_ft", batch_size=2, seq_len=512
        )
        want_lora = ref.estimate_memory_footprint(
            cfg, mode="lora_bf16", lora_config=lora_cfg, batch_size=2, seq_len=512
        )

        try:
            got_full = estimate_memory_footprint(
                cfg, mode="full_ft", batch_size=2, seq_len=512
            )
        except Exception as e:
            return {
                "full_ft_match": 0.0,
                "lora_bf16_match": 0.0,
                "size_ratio": 1.0,
                "_note": f"full_ft error on cfg {i}: {e}",
            }

        try:
            got_lora = estimate_memory_footprint(
                cfg, mode="lora_bf16", lora_config=lora_cfg, batch_size=2, seq_len=512
            )
        except Exception as e:
            return {
                "full_ft_match": 0.0,
                "lora_bf16_match": 0.0,
                "size_ratio": 1.0,
                "_note": f"lora_bf16 error on cfg {i}: {e}",
            }

        if got_full == want_full:
            full_ok += 1

        if got_lora == want_lora:
            lora_ok += 1

        if got_full and got_lora and got_full.get("total_peak_bytes", 0) > 0:
            ratio = got_lora.get("total_peak_bytes", 0) / got_full["total_peak_bytes"]
            if ratio > max_ratio:
                max_ratio = ratio

    return {
        "full_ft_match": 1.0 if full_ok == total else 0.0,
        "lora_bf16_match": 1.0 if lora_ok == total else 0.0,
        "size_ratio": float(max_ratio),
    }
