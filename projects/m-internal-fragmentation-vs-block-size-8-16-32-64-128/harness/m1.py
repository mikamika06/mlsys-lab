import ref
import random

def check(workdir):
    try:
        from kvblocks.metrics import compute_fragmentation
    except ImportError:
        return {"fragmentation_matches": 0.0}

    random.seed(42)
    seq_lens = [random.randint(1, 1000) for _ in range(50)]
    block_sizes = [8, 16, 32, 64, 128]

    want = ref.compute_fragmentation(seq_lens, block_sizes)
    try:
        got = compute_fragmentation(seq_lens, block_sizes)
    except NotImplementedError:
        return {"fragmentation_matches": 0.0}

    return {"fragmentation_matches": 1.0 if got == want else 0.0}
