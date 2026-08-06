import os
import sys

# Ensure parent and root directories are in sys.path so 'harness' can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
for path in [current_dir, os.path.abspath(os.path.join(current_dir, "..")), os.path.abspath(os.path.join(current_dir, "../.."))]:
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
from harness import Milestone, BenchmarkHarness

def column_parallel_matmul(x, w_col):
    raise NotImplementedError("Implement column_parallel_matmul")

def row_parallel_matmul(x_col, w_row):
    raise NotImplementedError("Implement row_parallel_matmul")

def full_parallel_matmul(x, w_col, w_row):
    raise NotImplementedError("Implement full_parallel_matmul")

harness = BenchmarkHarness("m-column-row-parallel-matmul-from-scratch")

@harness.milestone(1)
def milestone_1():
    x = torch.randn(32, 64)
    w_col = torch.randn(64, 16)
    res = column_parallel_matmul(x, w_col)
    expected = torch.matmul(x, w_col)
    assert torch.allclose(res, expected, atol=1e-5)
    return True

@harness.milestone(2)
def milestone_2():
    x_col = torch.randn(32, 32)
    w_row = torch.randn(32, 64)
    res = row_parallel_matmul(x_col, w_row)
    expected = torch.matmul(x_col, w_row)
    assert torch.allclose(res, expected, atol=1e-5)
    return True

@harness.milestone(3)
def milestone_3():
    x = torch.randn(32, 64)
    w_col = torch.randn(64, 32)
    w_row = torch.randn(32, 128)
    res = full_parallel_matmul(x, w_col, w_row)
    expected = torch.matmul(torch.matmul(x, w_col), w_row)
    assert torch.allclose(res, expected, atol=1e-5)
    return True

if __name__ == "__main__":
    harness.run()
