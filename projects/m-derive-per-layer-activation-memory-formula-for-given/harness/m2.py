import ref

def check(workdir):
    from actmem.crossover import find_attention_mlp_crossover
    matched = 1
    for cfg in ref.CONFIGS:
        want = ref.find_attention_mlp_crossover(cfg["h"], cfg["heads"], cfg["intermediate_size"])
        got = find_attention_mlp_crossover(cfg["h"], cfg["heads"], cfg["intermediate_size"])
        if got != want:
            matched = 0
            break
    return {"crossover_matched": float(matched)}
