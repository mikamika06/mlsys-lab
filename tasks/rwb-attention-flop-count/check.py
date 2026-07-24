def _oracle(batch, heads, seqlen_q, seqlen_k, head_dim, causal):
    qk = seqlen_q * seqlen_k * head_dim
    pv = seqlen_q * seqlen_k * head_dim
    flops = 2 * batch * heads * (qk + pv)
    if causal:
        flops //= 2
    return flops


def grade(sol, fx) -> dict:
    cases = [
        (1, 1, 16, 16, 64, False),
        (2, 8, 1024, 1024, 64, False),
        (4, 16, 512, 512, 128, True),
        (3, 12, 128, 256, 80, False),
        (2, 4, 77, 77, 32, True),
    ]
    ok = 1.0
    for case in cases:
        try:
            got = sol.attention_flops(*case)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(*case):
            ok = 0.0
            break
    return {"exact_match": ok}
