import numpy as np

def build_arguments(requests):
    cu_seqlens = [0]
    max_seqlen = 0
    tokens = []
    for req in requests:
        seq = req["tokens"]
        tokens.extend(seq)
        seql = len(seq)
        max_seqlen = max(max_seqlen, seql)
        cu_seqlens.append(cu_seqlens[-1] + seql)
    return {
        "tokens": np.array(tokens, dtype=np.int32),
        "cu_seqlens": np.array(cu_seqlens, dtype=np.int32),
        "max_seqlen": int(max_seqlen),
        "batch_size": len(requests)
    }
