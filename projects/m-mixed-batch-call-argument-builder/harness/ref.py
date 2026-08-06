import numpy as np

REQUESTS_LIST = [
    [{"tokens": [1, 2]}, {"tokens": [3, 4, 5]}],
    [{"tokens": [10]}],
    [{"tokens": [1, 2, 3, 4]}, {"tokens": [5, 6]}],
    [{"tokens": [7, 8, 9]}, {"tokens": [10, 11]}, {"tokens": [12]}],
    [{"tokens": [100, 101, 102, 103, 104]}]
]

HARDWARE = {"bandwidth": 2048.0, "flops": 32768.0}

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

def compute_offsets(batch_meta):
    cu = batch_meta["cu_seqlens"]
    offsets = []
    for i in range(len(cu) - 1):
        offsets.append((cu[i], cu[i+1]))
    return np.array(offsets, dtype=np.int32)

def analytic_latency(batch_meta, hardware_specs):
    bs = batch_meta["batch_size"]
    max_len = batch_meta["max_seqlen"]
    bw = hardware_specs["bandwidth"]
    flop = hardware_specs["flops"]
    ttft = (max_len * bs * 1024) / flop + 0.001
    itl = (bs * 512) / bw + 0.0005
    return {"ttft": float(ttft), "itl": float(itl)}
