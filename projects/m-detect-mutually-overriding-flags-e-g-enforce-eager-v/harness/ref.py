def detect_conflicts(args):
    out = []
    if args.get("enforce_eager") and args.get("cudagraph_capture_sizes"):
        out.append("enforce_eager conflicts with cudagraph_capture_sizes")
    if args.get("disable_sliding_window") and args.get("sliding_window"):
        out.append("disable_sliding_window conflicts with sliding_window")
    if args.get("kv_cache_dtype") == "fp8" and args.get("enable_prefix_caching") and args.get("no_prefix_caching"):
        out.append("enable_prefix_caching conflicts with no_prefix_caching")
    return out


def cli_to_engine(cli_args):
    mapping = {
        "model": "model",
        "enforce_eager": "enforce_eager",
        "max_model_len": "max_model_len",
        "max_num_batched_tokens": "max_num_batched_tokens",
        "gpu_memory_utilization": "gpu_memory_utilization",
    }
    return {mapping[k]: v for k, v in cli_args.items() if k in mapping}


def engine_to_cli(engine_kwargs):
    mapping = {
        "model": "model",
        "enforce_eager": "enforce_eager",
        "max_model_len": "max_model_len",
        "max_num_batched_tokens": "max_num_batched_tokens",
        "gpu_memory_utilization": "gpu_memory_utilization",
    }
    return {mapping[k]: v for k, v in engine_kwargs.items() if k in mapping}


def find_clamping_arg(requested_tokens, config):
    limit = config.get("max_model_len", 4096) * config.get("max_num_seqs", 256)
    if requested_tokens > limit:
        return "max_num_seqs or max_model_len"
    if "block_size" in config and requested_tokens % config["block_size"] != 0:
        return "block_size alignment"
    return None


CONFLICT_CASES = [
    ({"enforce_eager": True, "cudagraph_capture_sizes": [1, 2, 4]}, ["enforce_eager conflicts with cudagraph_capture_sizes"]),
    ({"disable_sliding_window": True, "sliding_window": 4096}, ["disable_sliding_window conflicts with sliding_window"]),
    ({"enforce_eager": False, "cudagraph_capture_sizes": [1, 2, 4]}, []),
]

TRANSLATION_CASES = [
    ({"model": "facebook/opt-125m", "enforce_eager": True, "max_model_len": 2048},
     {"model": "facebook/opt-125m", "enforce_eager": True, "max_model_len": 2048}),
    ({"model": "gpt2", "gpu_memory_utilization": 0.9, "max_num_batched_tokens": 8192},
     {"model": "gpt2", "gpu_memory_utilization": 0.9, "max_num_batched_tokens": 8192}),
]

CLAMP_CASES = [
    (1000000, {"max_model_len": 2048, "max_num_seqs": 16}, "max_num_seqs or max_model_len"),
    (513, {"block_size": 16, "max_model_len": 2048, "max_num_seqs": 256}, "block_size alignment"),
]
