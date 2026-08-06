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
