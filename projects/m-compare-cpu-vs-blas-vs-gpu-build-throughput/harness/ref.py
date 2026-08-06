import random

def get_throughput_fixtures():
    random.seed(42)
    return [
        {"backend": "cpu", "build_time_s": 120.0, "tokens_per_sec": 14.5},
        {"backend": "blas", "build_time_s": 145.0, "tokens_per_sec": 38.2},
        {"backend": "gpu", "build_time_s": 210.0, "tokens_per_sec": 112.4}
    ]

def get_size_analysis_fixtures():
    return {
        "feature": "Q4_0_4_4",
        "size_before_bytes": 104857600,
        "size_after_bytes": 104857600,
        "reason": "aliased_to_generic_block_structure_and_padded_alignment"
    }

def get_cmake_flags_fixtures():
    return {
        "cpu": ["-DGGML_STATIC=ON"],
        "blas": ["-DGGML_OPENBLAS=ON", "-DBLAS_LIBRARIES=/usr/lib"],
        "gpu": ["-DGGML_CUDA=ON", "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"]
    }
