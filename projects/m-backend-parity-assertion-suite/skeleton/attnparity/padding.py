def compute_attention(q, k, v, mask=None, backend="eager", is_causal=False):
    """Computes attention outputs for given inputs and backend."""
    raise NotImplementedError


def reproduce_right_padding_drift(samples, backend="eager", is_causal=True):
    """Reproduces attention numerical drift under right-padding."""
    raise NotImplementedError
