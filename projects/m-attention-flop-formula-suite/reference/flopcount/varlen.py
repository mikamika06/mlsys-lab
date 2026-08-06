from flopcount.attention import count_attention_flops


def count_varlen_attention_flops(
    seq_lens: list[int],
    h_q: int,
    d: int,
    causal: bool = True,
) -> int:
    """Calculates total attention FLOPs for a packed batch of variable sequence lengths."""
    total = 0
    for s in seq_lens:
        total += count_attention_flops(
            b=1, h_q=h_q, h_kv=h_q, s_q=s, s_k=s, d=d, causal=causal
        )
    return total


def flops_from_histogram(
    hist: dict[int, int],
    h_q: int,
    d: int,
    causal: bool = True,
) -> int:
    """Calculates total attention FLOPs given a sequence length histogram."""
    total = 0
    for s, count in hist.items():
        flops = count_attention_flops(
            b=1, h_q=h_q, h_kv=h_q, s_q=s, s_k=s, d=d, causal=causal
        )
        total += flops * count
    return total
