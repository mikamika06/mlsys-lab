import ref


def check(workdir):
    from efficiency.memory import measure_memory_footprints, rank_memory_usage

    out = {"ranking_match": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        mems = measure_memory_footprints(cfg)
        ranking = rank_memory_usage(mems)
        if ranking == ["lora_4bit", "lora_bf16", "full_ft"]:
            ok += 1

    if ok == len(ref.CONFIGS):
        out["ranking_match"] = 1.0
    else:
        out["_note"] = f"Memory ranking order did not match expected ['lora_4bit', 'lora_bf16', 'full_ft']"
    return out
