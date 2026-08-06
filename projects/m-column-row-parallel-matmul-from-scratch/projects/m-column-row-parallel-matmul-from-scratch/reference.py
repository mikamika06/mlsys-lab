import os
import sys

# Ensure parent and root directories are in sys.path so 'harness' can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
for path in [current_dir, os.path.abspath(os.path.join(current_dir, "..")), os.path.abspath(os.path.join(current_dir, "../.."))]:
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch.distributed as dist
from harness import Milestone, BenchmarkHarness

# Implement column-parallel and row-parallel matrix multiplication
def column_parallel_matmul(x, w_col):
    # x: [M, K], w_col: [K, N / world_size]
    # Local matrix multiplication yielding local output slice
    return torch.matmul(x, w_col)

def row_parallel_matmul(x_col, w_row):
    # x_col: [M, K / world_size], w_row: [K / world_size, N]
    # Local matrix multiplication followed by All-Reduce sum
    local_out = torch.matmul(x_col, w_row)
    if dist.is_initialized():
        dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
    return local_out

def full_parallel_matmul(x, w_col, w_row):
    # Two-layer tensor parallel matmul sequence (Column-Parallel -> Row-Parallel)
    hidden = column_parallel_matmul(x, w_col)
    out = row_parallel_matmul(hidden, w_row)
    return out

harness = BenchmarkHarness("m-column-row-parallel-matmul-from-scratch")

@harness.milestone(1)
def milestone_1():
    # Milestone 1: Column Parallel Matmul verification
    x = torch.randn(32, 64)
    w_col = torch.randn(64, 16)
    res = column_parallel_matmul(x, w_col)
    expected = torch.matmul(x, w_col)
    assert torch.allclose(res, expected, atol=1e-5), "Column-parallel matmul mismatch"
    return True

@harness.milestone(2)
def milestone_2():
    # Milestone 2: Row Parallel Matmul verification
    x_col = torch.randn(32, 32)
    w_row = torch.randn(32, 64)
    res = row_parallel_matmul(x_col, w_row)
    expected = torch.matmul(x_col, w_row)
    assert torch.allclose(res, expected, atol=1e-5), "Row-parallel matmul mismatch"
    return True

@harness.milestone(3)
def milestone_3():
    # Milestone 3: End-to-end Column-Row Parallel Matmul sequence
    x = torch.randn(32, 64)
    w_col = torch.randn(64, 32)
    w_row = torch.randn(32, 128)
    res = full_parallel_matmul(x, w_col, w_row)
    expected = torch.matmul(torch.matmul(x, w_col), w_row)
    assert torch.allclose(res, expected, atol=1e-5), "Full sequence matmul mismatch"
    return True

if __name__ == "__main__":
    harness.run()
