import math
import random
from mlsys import scorers

def _ref_best(token_sizes, table_overhead_per_block):
    candidates = [16, 32, 64, 128, 256, 512, 1024]
    best_b = None
    best_ratio = -1.0
    total_useful = sum(token_sizes)
    for b in candidates:
        total_blocks = 0
        total_alloc = 0
        for s in token_sizes:
            blocks = math.ceil(s / b)
            total_blocks += blocks
            total_alloc += blocks * b
        total_alloc += total_blocks * table_overhead_per_block
        ratio = total_useful / total_alloc
        if ratio > best_ratio + 1e-12 or (abs(ratio - best_ratio) < 1e-12 and (best_b is None or b < best_b)):
            best_ratio = ratio
            best_b = b
    return best_b, best_ratio


def grade(sol, fx) -> dict:
    random.seed(123)
    size_ratio_acc = []
    for _ in range(5):
        token_sizes = [random.randint(8, 1500) for _ in range(200)]
        overhead = random.choice([8, 16, 24])
        ref_b, ref_ratio = _ref_best(token_sizes, overhead)
        try:
            got_b, got_ratio = sol.choose_kv_block_size(token_sizes, overhead)
        except Exception:
            return {"size_ratio": 0.0}
        if not isinstance(got_b, int) or not isinstance(got_ratio, (int, float)):
            return {"size_ratio": 0.0}
        # compute realized ratio: if same block, got_ratio = ref_ratio; otherwise recompute for their b
        if got_b in [16, 32, 64, 128, 256, 512, 1024]:
            total_useful = sum(token_sizes)
            total_blocks = sum(math.ceil(s / got_b) for s in token_sizes)
            total_alloc = total_blocks * got_b + total_blocks * overhead
            actual_ratio = total_useful / total_alloc
        else:
            actual_ratio = 0.0
        size_ratio_acc.append(actual_ratio / ref_ratio)
    return {"size_ratio": float(sum(size_ratio_acc) / len(size_ratio_acc))}
