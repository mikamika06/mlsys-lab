import random

def generate_test_cases():
    defaults = {"port": 8000, "tensor_parallel_size": 1, "max_model_len": 4096, "gpu_memory_utilization": 0.9}
    yaml_cfg = {"port": 9000, "max_model_len": 2048}
    env_cfg = {"port": 7000, "tensor_parallel_size": 4}
    cli_cfg = {"port": 6000}

    resolved = dict(defaults)
    for d in [yaml_cfg, env_cfg, cli_cfg]:
        for k, v in d.items():
            if v is not None:
                resolved[k] = v

    argv = ["--port=8000", "--host", "localhost", "--enable-prefix-caching"]
    parsed_argv = {"port": "8000", "host": "localhost", "enable-prefix-caching": True}

    all_args = [f"arg_{i}" for i in range(40)]
    return {
        "defaults": defaults,
        "yaml_cfg": yaml_cfg,
        "env_cfg": env_cfg,
        "cli_cfg": cli_cfg,
        "resolved": resolved,
        "argv": argv,
        "parsed_argv": parsed_argv,
        "all_args": all_args
    }
