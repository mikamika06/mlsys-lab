from benchrunner.benchmark import attribute_runner_delta
from benchrunner.selector import select_runner
from benchrunner.portability import diagnose_portability_failure


BENCH_CASES = [
    {
        "gen_tokens": 100,
        "llamafile": {
            "gen_ms": 1000.0,
            "ipc_overhead_ms": 2.0,
            "threads": 8,
            "isa_features": ["AVX2", "FMA", "AVX512F"],
        },
        "ollama": {
            "gen_ms": 2000.0,
            "ipc_overhead_ms": 5.0,
            "threads": 8,
            "isa_features": ["AVX2", "FMA"],
        },
    },
    {
        "gen_tokens": 200,
        "llamafile": {
            "gen_ms": 2000.0,
            "ipc_overhead_ms": 3.0,
            "threads": 8,
            "isa_features": ["AVX2", "FMA"],
        },
        "ollama": {
            "gen_ms": 2500.0,
            "ipc_overhead_ms": 25.0,
            "threads": 8,
            "isa_features": ["AVX2", "FMA"],
        },
    },
    {
        "gen_tokens": 150,
        "llamafile": {
            "gen_ms": 1500.0,
            "ipc_overhead_ms": 2.0,
            "threads": 8,
            "isa_features": ["AVX2"],
        },
        "ollama": {
            "gen_ms": 2000.0,
            "ipc_overhead_ms": 4.0,
            "threads": 4,
            "isa_features": ["AVX2"],
        },
    },
    {
        "gen_tokens": 100,
        "llamafile": {
            "gen_ms": 1000.0,
            "ipc_overhead_ms": 2.0,
            "threads": 8,
            "isa_features": ["AVX2"],
        },
        "ollama": {
            "gen_ms": 1000.0,
            "ipc_overhead_ms": 2.0,
            "threads": 8,
            "isa_features": ["AVX2"],
        },
    },
]

SELECTOR_CASES = [
    {
        "single_file_dist": True,
        "air_gapped": True,
        "os_targets": ["linux", "windows", "darwin"],
        "api_standard": "cli_embedded",
        "multi_model_serving": False,
    },
    {
        "single_file_dist": False,
        "air_gapped": False,
        "os_targets": ["linux"],
        "api_standard": "openai_http",
        "multi_model_serving": True,
    },
]

PORTABILITY_CASES = [
    (
        {
            "arch": "x86_64",
            "cpu_flags": ["AVX", "AVX2"],
            "page_size_kb": 4,
            "kernel_version": "5.15.0",
        },
        {
            "supported_arches": ["x86_64", "arm64"],
            "required_cpu_flags": ["AVX", "AVX2", "AVX512F"],
            "supported_page_sizes_kb": [4, 64],
            "min_kernel_version": "5.4.0",
        },
    ),
    (
        {
            "arch": "riscv64",
            "cpu_flags": ["V"],
            "page_size_kb": 4,
            "kernel_version": "5.15.0",
        },
        {
            "supported_arches": ["x86_64", "arm64"],
            "required_cpu_flags": [],
            "supported_page_sizes_kb": [4],
            "min_kernel_version": "5.0.0",
        },
    ),
    (
        {
            "arch": "x86_64",
            "cpu_flags": ["AVX", "AVX2"],
            "page_size_kb": 16,
            "kernel_version": "5.15.0",
        },
        {
            "supported_arches": ["x86_64"],
            "required_cpu_flags": ["AVX"],
            "supported_page_sizes_kb": [4, 64],
            "min_kernel_version": "5.0.0",
        },
    ),
    (
        {
            "arch": "x86_64",
            "cpu_flags": ["AVX", "AVX2"],
            "page_size_kb": 4,
            "kernel_version": "4.19.0",
        },
        {
            "supported_arches": ["x86_64"],
            "required_cpu_flags": ["AVX"],
            "supported_page_sizes_kb": [4],
            "min_kernel_version": "5.4.0",
        },
    ),
]
