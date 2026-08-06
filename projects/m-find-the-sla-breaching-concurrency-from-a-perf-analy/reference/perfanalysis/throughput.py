def compute_tokens_metrics(fixture):
    t = fixture["total_time_sec"]
    tok = fixture["total_tokens"]
    gpus = fixture["gpu_count"]
    users = fixture["active_users"]
    return {
        "tokens_per_sec_gpu": (tok / t) / gpus,
        "tokens_per_sec_user": (tok / t) / users
    }
