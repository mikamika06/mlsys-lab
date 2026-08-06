CONFIGS = [
    {"model_path": "meta-llama/Llama-3-8B-Instruct", "port": 30000, "disable_radix_cache": False, "extra_args": None},
    {"model_path": "meta-llama/Llama-3-8B-Instruct", "port": 30001, "disable_radix_cache": True, "extra_args": {"--tp-size": 2}},
    {"model_path": "mistralai/Mistral-7B-Instruct-v0.2", "port": 8000, "disable_radix_cache": False, "extra_args": {"--mem-fraction-static": 0.8}}
]


def build_launch_command(model_path, port=30000, disable_radix_cache=False, extra_args=None):
    cmd = ["python", "-m", "sglang.launch_server", "--model-path", str(model_path), "--port", str(port)]
    if disable_radix_cache:
        cmd.append("--disable-radix-cache")
    if extra_args:
        for k, v in extra_args.items():
            cmd.extend([str(k), str(v)])
    return cmd


def compute_latency_ratio(ttft_enabled, ttft_disabled):
    if ttft_disabled <= 0:
        return 0.0
    return float(ttft_enabled) / float(ttft_disabled)
