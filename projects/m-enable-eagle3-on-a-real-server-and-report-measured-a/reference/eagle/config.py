def build_eagle_config(cfg):
    return {
        "model_name": cfg["model"],
        "speculative_model": f"{cfg['model']}-eagle3",
        "speculative_draft_tensor_parallel_size": 1,
        "num_speculative_tokens": cfg["num_speculative_tokens"],
        "speculative_max_model_len": 4096,
        "tree_config": {
            "top_k": cfg["top_k"],
            "depth": cfg["depth"]
        }
    }
