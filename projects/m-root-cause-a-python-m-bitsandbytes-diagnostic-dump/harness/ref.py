DUMPS = [
    (
        "CUDA version: 12.1\nPlatform: linux_x86_64\nLibrary path: /usr/local/lib\nError: libcudart.so not found\nSymbol: cudaMalloc",
        {
            "cuda_version": "12.1",
            "platform": "linux_x86_64",
            "lib_path": "/usr/local/lib",
            "error": "libcudart.so not found",
            "symbols": ["cudaMalloc"]
        }
    ),
    (
        "CUDA version: 11.8\nPlatform: linux_aarch64\nLibrary path: /opt/cuda\nError: CUDA version mismatch between runtime and driver\nSymbol: cublasCreate",
        {
            "cuda_version": "11.8",
            "platform": "linux_aarch64",
            "lib_path": "/opt/cuda",
            "error": "CUDA version mismatch between runtime and driver",
            "symbols": ["cublasCreate"]
        }
    ),
    (
        "CUDA version: 12.4\nPlatform: windows_x64\nLibrary path: C:\\Program Files\nError: shared object failed to initialize\n",
        {
            "cuda_version": "12.4",
            "platform": "windows_x64",
            "lib_path": "C:\\Program Files",
            "error": "shared object failed to initialize",
            "symbols": []
        }
    )
]

CLASSIFICATIONS = [
    ({"error": "libcudart.so not found"}, "MISSING_CUDART"),
    ({"error": "CUDA version mismatch between runtime and driver"}, "CUDA_VERSION_MISMATCH"),
    ({"error": "shared object failed to initialize"}, "MISSING_SHARED_OBJECT"),
    ({"error": ""}, "SUCCESS")
]

RESOLUTIONS = [
    ("12.1", "linux_x86_64", "libbitsandbytes_cuda12.so"),
    ("11.8", "linux_aarch64", "libbitsandbytes_cuda11_sbs.so"),
    ("10.2", "win_amd64", "libbitsandbytes_cpu.so")
]
