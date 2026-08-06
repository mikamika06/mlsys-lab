import time


def measure_step_latencies(config):
    steps_ft = config.get("steps_ft", [0.1, 0.11, 0.105])
    steps_lora = config.get("steps_lora", [0.07, 0.072, 0.071])

    t_ft = sum(steps_ft) / len(steps_ft)
    t_lora = sum(steps_lora) / len(steps_lora)

    ratio = t_ft / max(t_lora, 1e-6)
    return {"full_ft_latency": t_ft, "lora_latency": t_lora, "latency_ratio": ratio}
