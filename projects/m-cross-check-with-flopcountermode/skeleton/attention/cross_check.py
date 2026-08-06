from attention.ops import matmul, softmax
from attention.tracker import FlopCounterMode


def attention_forward(b, h, seq, d):
    """Simulate forward pass of standard attention."""
    raise NotImplementedError


def empirical_flops(b, h, seq, d):
    """Measures FLOPs dynamically."""
    raise NotImplementedError


def analytical_flops(b, h, seq, d):
    """Calculates theoretical FLOPs strictly by formula."""
    raise NotImplementedError


def rel_err(b, h, seq, d):
    """Computes relative error between empirical and analytical FLOPs."""
    raise NotImplementedError
