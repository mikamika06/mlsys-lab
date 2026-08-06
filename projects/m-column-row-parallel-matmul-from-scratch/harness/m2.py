import harness.ref as ref


def check(workdir):
    from tp.comm import estimate_tp_layer_comm_bytes

    out = {"volume_match": 0.0}

    test_cases = [
        (4, 128, 768, 3072, 1, 2),
        (2, 512, 1024, 4096, 2, 2),
        (8, 256, 2048, 8192, 4, 2),
        (1, 1024, 4096, 11008, 8, 2),
    ]

    matched = 0
    for b, s, h, ffn, tp, elem in test_cases:
        want = ref.compute_ref_comm(b, s, h, ffn, tp, elem)
        got = estimate_tp_layer_comm_bytes(b, s, h, ffn, tp, elem)
        if got == want:
            matched += 1
        else:
            out["_note"] = f"case (b={b}, s={s}, h={h}, tp={tp}): got {got}, want {want}"
            return out

    if matched == len(test_cases):
        out["volume_match"] = 1.0

    return out
