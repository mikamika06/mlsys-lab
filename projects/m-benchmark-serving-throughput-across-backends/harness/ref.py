def get_test_workloads():
    return [
        {"batch_size": 8, "prompt_len": 256, "gen_len": 64},
        {"batch_size": 16, "prompt_len": 512, "gen_len": 128},
        {"batch_size": 32, "prompt_len": 1024, "gen_len": 256},
    ]


def get_hardware_profiles():
    return [
        {"name": "NVIDIA H100 80GB HBM3", "cap": (9, 0), "is_hopper": True},
        {"name": "NVIDIA H800", "cap": (9, 0), "is_hopper": True},
        {"name": "NVIDIA A100-SXM4-80GB", "cap": (8, 0), "is_hopper": False},
        {"name": "NVIDIA RTX 4090", "cap": (8, 9), "is_hopper": False},
    ]
