def calculate_vram(model_config, offload_experts_to_cpu: bool):
    raise NotImplementedError


def max_context_length(vram_total_bytes, model_config, offload_experts_to_cpu: bool):
    raise NotImplementedError
