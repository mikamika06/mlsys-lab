def trtexec_to_config(args_dict):
    config = {
        "fp16": args_dict.get("fp16", False),
        "int8": args_dict.get("int8", False),
        "max_workspace_size": args_dict.get("memPoolSize", 1 << 30),
        "max_batch_size": args_dict.get("batch", 1),
    }
    return config

def build_engine(network, config):
    return {
        "network": network,
        "config": config,
        "serialized": True
    }
