import torch
from matmul.kernel import triton_matmul

def tune_matmul(a, b):
    return triton_matmul(a, b, group_size_m=8)
