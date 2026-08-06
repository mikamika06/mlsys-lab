def count_varlen_attention_flops(
    seq_lens: list[int],
    h_q: int,
    d: int,
    causal: bool = True,
) -> int:
    """Calculates total attention FLOPs for a packed batch of variable sequence lengths."""
    raise NotImplementedError


def flops_from_histogram(
    hist: dict[int, int],
    h_q: int,
    d: int,
    causal: bool = True,
) -> int:
    """Calculates total attention FLOPs given a sequence length histogram."""
    raise NotImplementedError
