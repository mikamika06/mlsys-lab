def translate_vllm_to_sglang(vllm_config):
    sgl = {}
    sgl["max_running_requests"] = vllm_config.get("max_num_seqs", 256)
    sgl["chunked_prefill_size"] = vllm_config.get("max_num_batched_tokens", 512)
    sgl["attention_backend"] = "flashinfer"
    return sgl
