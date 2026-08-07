import sys

sys.path.insert(0, ".")
from flex.block_mask import compute_block_mask_indices

def test_block_causality():
    sparsity, nums, idxs = compute_block_mask_indices(1000, 100, 32)
    for q_b in range(len(nums)):
        for i in range(nums[q_b]):
            kv_b = idxs[q_b, i]
            assert kv_b <= q_b, "Found KV block from the future"

def test_block_window_limit():
    seq, win, bsz = 1000, 100, 32
    sparsity, nums, idxs = compute_block_mask_indices(seq, win, bsz)
    for q_b in range(len(nums)):
        for i in range(nums[q_b]):
            kv_b = idxs[q_b, i]
            q_min = q_b * bsz
            kv_max = kv_b * bsz + bsz - 1
            assert kv_max > q_min - win, "Found KV block entirely outside window"
