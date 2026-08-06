def get_test_snippets():
    return [
        (["add rax, rbx", "vaddps ymm0, ymm1, ymm2"], "L0"),
        (["vaddps zmm0, zmm1, zmm2", "vpand zmm3, zmm4, zmm5"], "L1"),
        (["vfmadd231ps zmm0, zmm1, zmm2", "add rax, 1"], "L2"),
        (["vpdpbusd zmm0, zmm1, zmm2"], "L2"),
        (["vmovups zmm0, [rax]"], "L1"),
        (["vpmaddwd ymm0, ymm1, ymm2"], "L0"),
    ]


def get_simulation_cases():
    return [
        {
            "stream": [
                ("vpdpbusd zmm1, zmm2, zmm3", 10000),
                ("add rax, rbx", 40000),
                ("add rax, rbx", 20000),
            ],
            "base_freq_ghz": 3.0,
            "recovery_cycles": 50000,
        },
        {
            "stream": [
                ("vaddps zmm0, zmm1, zmm2", 20000),
                ("vaddps ymm0, ymm1, ymm2", 30000),
            ],
            "base_freq_ghz": 2.5,
            "recovery_cycles": 40000,
        },
        {
            "stream": [
                ("add rax, rbx", 100000),
            ],
            "base_freq_ghz": 3.2,
            "recovery_cycles": 50000,
        },
    ]


def get_vnni_cases():
    return [
        {
            "num_mac_ops": 12800000,
            "vnni_vector_width": 512,
            "fallback_vector_width": 256,
            "base_freq_ghz": 3.0,
            "recovery_cycles": 50000,
        },
        {
            "num_mac_ops": 64000000,
            "vnni_vector_width": 512,
            "fallback_vector_width": 512,
            "base_freq_ghz": 2.8,
            "recovery_cycles": 30000,
        },
        {
            "num_mac_ops": 25600000,
            "vnni_vector_width": 256,
            "fallback_vector_width": 256,
            "base_freq_ghz": 3.5,
            "recovery_cycles": 50000,
        },
    ]
