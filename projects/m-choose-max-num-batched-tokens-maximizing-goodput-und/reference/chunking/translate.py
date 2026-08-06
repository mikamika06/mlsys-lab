def translate_vllm_to_sglang(vllm_config):
    sglang_config = dict(vllm_config)
    if "max_num_batched_tokens" in sglang_config:
        val = sglang_config.pop("max_num_batched_tokens")
        sglang_config["chunked_prefill_size"] = val
    if sglang_config.get("chunked_prefill_size") == -1:
        sglang_config["chunked_prefill_size"] = 0
    return sglang_config
