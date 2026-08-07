import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from ftmem.memory import estimate_memory_footprint
    except Exception as e:
        return {
            "qlora_mem_match": 0.0,
            "qlora_saves_memory": 0.0,
            "_note": f"Import error: {e}",
        }

    qlora_ok = 0
    saves_count = 0
    total = len(ref.CONFIGS)

    for i, cfg in enumerate(ref.CONFIGS):
        lora_cfg = ref.LORA_CONFIGS[i % len(ref.LORA_CONFIGS)]

        want_qlora = ref.estimate_memory_footprint(
            cfg, mode="qlora_4bit", lora_config=lora_cfg, batch_size=2, seq_len=512
        )

        try:
            got_qlora = estimate_memory_footprint(
                cfg, mode="qlora_4bit", lora_config=lora_cfg, batch_size=2, seq_len=512
            )
            got_lora = estimate_memory_footprint(
                cfg, mode="lora_bf16", lora_config=lora_cfg, batch_size=2, seq_len=512
            )
        except Exception as e:
            return {
                "qlora_mem_match": 0.0,
                "qlora_saves_memory": 0.0,
                "_note": f"qlora_4bit error on cfg {i}: {e}",
            }

        if got_qlora == want_qlora:
            qlora_ok += 1

        if (
            got_qlora
            and got_lora
            and got_qlora.get("total_static_bytes", 0) < got_lora.get("total_static_bytes", 0)
            and got_qlora.get("base_weights_bytes", 0) < got_lora.get("base_weights_bytes", 0)
        ):
            saves_count += 1

    return {
        "qlora_mem_match": 1.0 if qlora_ok == total else 0.0,
        "qlora_saves_memory": 1.0 if saves_count == total else 0.0,
    }
