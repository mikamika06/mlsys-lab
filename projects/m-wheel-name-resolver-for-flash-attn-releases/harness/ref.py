WHEELS = [
    "flash_attn-2.5.8-cp310-cp310-linux_x86_64.whl",
    "flash_attn-2.5.7+cu122-cp311-cp311-linux_x86_64.whl",
    "flash_attn-2.5.6-cp312-cp312-win_amd64.whl",
    "flash_attn-2.5.5-cp39-cp39-manylinux2014_x86_64.whl",
    "flash_attn-2.5.4+cu121-cp310-cp310-linux_aarch64.whl",
    "flash_attn-2.5.3-py3-none-any.whl",
    "flash_attn-2.5.2+cu118-cp38-cp38-linux_x86_64.whl",
    "flash_attn-2.5.1-cp311-cp311-win_amd64.whl",
    "flash_attn-2.5.0-cp310-cp310-manylinux_2_17_x86_64.whl",
    "flash_attn-2.4.2-cp39-cp39-linux_x86_64.whl",
    "flash_attn-2.4.1+cu121-cp312-cp312-linux_x86_64.whl",
    "flash_attn-2.4.0-cp310-cp310-linux_x86_64.whl",
]

ENVIRONMENTS = [
    {"py": "3.10", "cu": "12.1", "torch": "2.2.0"},
    {"py": "3.11", "cu": "12.2", "torch": "2.2.0"},
    {"py": "3.12", "cu": "12.1", "torch": "2.3.0"},
    {"py": "3.9", "cu": "11.8", "torch": "2.1.0"},
    {"py": "3.10", "cu": "12.1", "torch": "2.2.0"},
    {"py": "3.10", "cu": "11.8", "torch": "2.0.0"},
    {"py": "3.8", "cu": "11.8", "torch": "2.0.0"},
    {"py": "3.11", "cu": "12.1", "torch": "2.1.0"},
    {"py": "3.10", "cu": "12.1", "torch": "2.2.0"},
    {"py": "3.9", "cu": "11.8", "torch": "2.1.0"},
    {"py": "3.12", "cu": "12.1", "torch": "2.3.0"},
    {"py": "3.10", "cu": "12.1", "torch": "2.2.0"},
]

TRACEBACKS = [
    ("ERROR: flash_attn-2.5.8 is not a supported wheel on this platform.", "wheel_tag_mismatch"),
    ("RuntimeError: CUDA error: no kernel image is available for execution on device", "cuda_version_mismatch"),
    ("ImportError: /lib/python3.10/site-packages/flash_attn_2_cuda.so: undefined symbol: _ZN3c104Error...", "abi_incompatibility"),
    ("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB", "out_of_memory"),
    ("Ninja build stopped: subcommand failed. error: command 'gcc' failed with exit status 1", "build_compilation_failure"),
    ("ValueError: Wheel filename has invalid format", "unknown_error"),
    ("pip._internal.exceptions.UnsupportedWheel: flash_attn-2.5.5-cp39-cp39-win.whl is not a supported wheel", "wheel_tag_mismatch"),
    ("RuntimeError: nvcc fatal : Unsupported gpu architecture 'compute_90'", "cuda_version_mismatch"),
    ("ImportError: undefined symbol: PyModule_Create2", "abi_incompatibility"),
    ("OutOfMemoryError: CUDA out of memory during flash attention forward pass", "out_of_memory"),
    ("Failed to build flash-attn; ninja: build stopped due to errors", "build_compilation_failure"),
    ("Some random unexpected error string without patterns", "unknown_error"),
]
