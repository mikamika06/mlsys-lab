def compare_servers(lm_res, mlx_res):
    text_match = lm_res["text"].strip().rstrip("!") == mlx_res["text"].strip().rstrip("!")
    token_match = lm_res["tokens"] == mlx_res["tokens"]
    latency_ratio = mlx_res["latency_ms"] / lm_res["latency_ms"]
    return {
        "text_match": text_match,
        "token_match": token_match,
        "latency_ratio": float(latency_ratio)
    }
