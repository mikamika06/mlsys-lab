import ref

def check(workdir):
    from actmem.formula import layer_activation_memory

    test_cases = [
        (1, 512, 2048, 16, 2),
        (2, 1024, 4096, 32, 4),
        (4, 2048, 1024, 8, 2)
    ]
    ok = 0
    for tc in test_cases:
        want = ref.layer_activation_memory(*tc)
        got = layer_activation_memory(*tc)
        if got == want:
            ok += 1
    return {"formulas_matched": float(ok)}
