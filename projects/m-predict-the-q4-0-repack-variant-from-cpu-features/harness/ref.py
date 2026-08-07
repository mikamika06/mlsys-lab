import numpy as np

TEST_CASES = [
    ({"sve": True}, "q4_0_sve"),
    ({"avx512f": True, "avx2": True}, "q4_0_avx512"),
    ({"avx2": True}, "q4_0_avx2"),
    ({"neon": True}, "q4_0_neon"),
    ({}, "q4_0_scalar"),
    ({"avx512f": False, "avx2": False, "neon": False}, "q4_0_scalar")
]

SAMPLE_WEIGHTS = np.random.default_rng(123).random(1024).astype(np.float32)

SAMPLE_LOG = "gcc -O3 main.c\nerror: undefined reference to 'ggml_vec_dot_q4_0'\nmake: *** [all] Error 2"
