import random

def get_test_cases():
    rng = random.Random(42)
    tensors_good = {"blk.0.attn_q.weight": [2048, 2048], "output.weight": [32000, 2048]}
    imatrix_good = {
        "blk.0.attn_q.weight": {"shape": [2048, 2048]},
        "output.weight": {"shape": [32000, 2048]}
    }
    tensors_bad = {"blk.0.attn_q.weight": [2048, 2048], "output.weight": [32000, 2048]}
    imatrix_bad = {
        "blk.0.attn_q.weight": {"shape": [1024, 2048]},
        "output.weight": {"shape": [32000, 2048]}
    }
    return tensors_good, imatrix_good, tensors_bad, imatrix_bad
