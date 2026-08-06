import sys
sys.path.insert(0, ".")
from dispatch.selector import dispatch_kernel, is_eligible


def test_dispatch_selection_invariants():
    kernels = [
        {
            "name": "fast_fp16_gemm",
            "priority": 100,
            "allowed_in_dtypes": ["float16"],
            "allowed_out_dtypes": ["float16"],
            "quant_scheme": "none",
            "min_k": 64,
            "align_k": 16,
            "align_n": 16,
            "req_align_bytes": 16,
        },
        {
            "name": "fast_awq_int4",
            "priority": 90,
            "allowed_in_dtypes": ["float16"],
            "allowed_out_dtypes": ["float16"],
            "quant_scheme": "awq_int4",
            "group_size": 128,
            "min_k": 128,
            "align_k": 64,
            "align_n": 64,
            "req_align_bytes": 16,
        },
    ]

    cfg_valid = {
        "in_dtype": "float16",
        "out_dtype": "float16",
        "quant_scheme": "none",
        "k": 128,
        "n": 128,
        "ptr_align_bytes": 16,
    }
    assert dispatch_kernel(kernels, cfg_valid) == "fast_fp16_gemm"

    cfg_unaligned = dict(cfg_valid, k=130)
    assert not is_eligible(kernels[0], cfg_unaligned)
    assert dispatch_kernel(kernels, cfg_unaligned) == "fallback_gemm"

    cfg_awq = {
        "in_dtype": "float16",
        "out_dtype": "float16",
        "quant_scheme": "awq_int4",
        "group_size": 128,
        "k": 256,
        "n": 256,
        "ptr_align_bytes": 16,
    }
    assert dispatch_kernel(kernels, cfg_awq) == "fast_awq_int4"
