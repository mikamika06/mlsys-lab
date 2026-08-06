def enumerate_flags(backend):
    table = {
        "cpu": ["-DGGML_STATIC=ON"],
        "blas": ["-DGGML_OPENBLAS=ON", "-DBLAS_LIBRARIES=/usr/lib"],
        "gpu": ["-DGGML_CUDA=ON", "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"]
    }
    if backend not in table:
        raise ValueError("unknown backend")
    return sorted(table[backend])
