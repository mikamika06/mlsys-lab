CPU_SAMPLES = [
    {
        "code": "#pragma omp parallel for\nfor(int i=0; i<100; ++i) {\n  #pragma ivdep\n  a[i] = b[i];\n}",
        "expected": {"has_openmp": True, "has_vectorized": True, "loop_count": 1}
    },
    {
        "code": "for(int j=0; j<50; ++j) { c[j] = 0; }",
        "expected": {"has_openmp": False, "has_vectorized": False, "loop_count": 1}
    },
    {
        "code": "#pragma omp parallel\n{ for(int k=0; k<10; ++k) {} }",
        "expected": {"has_openmp": True, "has_vectorized": False, "loop_count": 1}
    }
]

TRITON_SAMPLES = [
    {
        "dump": "BLOCK_SIZE = 128\nnum_warps = 4\ngrid = [256, 1]",
        "expected": {"block_size": 128, "num_warps": 4, "grid": 256}
    },
    {
        "dump": "# BLOCK_SIZE: 64, num_warps: 8, grid: [512]",
        "expected": {"block_size": 64, "num_warps": 8, "grid": 512}
    },
    {
        "dump": "meta: BLOCK_SIZE=256, num_warps=2, grid=[1024]",
        "expected": {"block_size": 256, "num_warps": 2, "grid": 1024}
    }
]

MLP_CONFIGS = [
    [{"fused": True, "layers": 2}, {"fused": False, "layers": 3}],
    [{"fused": True, "layers": 4}]
]
