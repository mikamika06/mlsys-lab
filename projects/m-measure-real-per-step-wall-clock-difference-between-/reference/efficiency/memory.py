def measure_memory_footprints(config):
    base_mem = config.get("base_memory", 1000.0)
    return {
        "full_ft": base_mem * 4.0,
        "lora_bf16": base_mem * 1.8,
        "lora_4bit": base_mem * 1.1
    }


def rank_memory_usage(measurements):
    sorted_items = sorted(measurements.items(), key=lambda x: x[1])
    return [k for k, v in sorted_items]
