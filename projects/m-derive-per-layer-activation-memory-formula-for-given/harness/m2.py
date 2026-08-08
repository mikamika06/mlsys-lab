import ref

def check(workdir):
    from actmem.crossover import find_attention_mlp_crossover

    tc = (1, 4096, 32, 2)
    want = ref.find_attention_mlp_crossover(*tc)
    got = find_attention_mlp_crossover(*tc)
    matched = 1.0 if got == want else 0.0
    return {"crossovers_matched": matched}
