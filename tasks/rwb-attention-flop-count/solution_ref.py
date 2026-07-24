def attention_flops(batch, heads, seqlen_q, seqlen_k, head_dim, causal):
    qk_flops = 2 * batch * heads * seqlen_q * seqlen_k * head_dim
    pv_flops = 2 * batch * heads * seqlen_q * seqlen_k * head_dim
    total = qk_flops + pv_flops
    if causal:
        total //= 2
    return total
