import ref


def check(workdir):
    from hybridkv.memory import effective_bytes_per_token
    ok = 0
    tests = [(cfg, seq) for cfg in ref.CONFIGS for seq in [64, 128, 512, 1024]]
    for cfg, seq in tests:
        want = ref.effective_bytes_per_token(cfg, seq)
        got = effective_bytes_per_token(cfg, seq)
        if want == got:
            ok += 1
    out = {"effective_bytes_match": 1.0 if ok == len(tests) else 0.0}
    return out
