import numpy as np
from sparse.masks import generate_nm_mask
from sparse.error import capture_sparse_matmul_error
from sparse.bench import parse_a100_gemm_log

SAMPLE_LOG = """
kernel: gemm_s8 | dense_tflops: 120.5 TFLOPS | sparse_tflops: 235.0 TFLOPS | speedup: 1.95x
kernel: gemm_fp16 | dense_tflops: 150.0 TFLOPS | sparse_tflops: 290.0 TFLOPS | speedup: 1.93x
"""
