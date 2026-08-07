def get_sample_config():
    return {"context_length": 1024, "num_layers": 16, "hidden_size": 2048, "bytes_per_param": 2}

def get_large_configs():
    return [
        {"context_length": 16384, "num_layers": 32, "hidden_size": 4096, "bytes_per_param": 2},
        {"context_length": 32768, "num_layers": 32, "hidden_size": 4096, "bytes_per_param": 2},
        {"context_length": 65536, "num_layers": 32, "hidden_size": 4096, "bytes_per_param": 2}
    ]
