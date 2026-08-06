from attention.ops import matmul, softmax
from attention.tracker import FlopCounterMode


def attention_forward(b, h, seq, d):
    """Simulate forward pass of standard attention."""
    scores = matmul((b, h, seq, d), (b, h, d, seq))
    probs = softmax(scores)
    out = matmul(probs, (b, h, seq, d))
    return out


def empirical_flops(b, h, seq, d):
    """Measures FLOPs dynamically."""
    with FlopCounterMode() as counter:
        attention_forward(b, h, seq, d)
    return counter.total


def analytical_flops(b, h, seq, d):
    """Calculates theoretical FLOPs strictly by formula."""
    macs_qk = b * h * seq * seq * d
    macs_sv = b * h * seq * seq * d
    ops_softmax = 5 * b * h * seq * seq
    return 2 * (macs_qk + macs_sv) + ops_softmax


def rel_err(b, h, seq, d):
    """Computes relative error between empirical and analytical FLOPs."""
    emp = empirical_flops(b, h, seq, d)
    ana = analytical_flops(b, h, seq, d)
    return abs(emp - ana) / emp
