import ref
from fa_dispatch.backends import dispatch_check

def execute_attention(q, k, v, mask=None):
    valid, backend = dispatch_check(q, k, v, mask)
    if not valid:
        raise RuntimeError("Silent fallback detected: conditions not met for fast path")
    return ref.compute_attention_fast(q, k, v, mask, backend)
