import numpy as np


def measure_peak_memory(weights, total_inputs, total_targets, accumulation_steps):
    total_samples = len(total_inputs)
    micro_batch_size = total_samples // accumulation_steps

    weights_mem = weights.size * 8
    grad_mem = weights.size * 8
    micro_input_mem = micro_batch_size * total_inputs.shape[1] * 8
    micro_target_mem = micro_batch_size * total_targets.shape[1] * 8
    activation_mem = micro_batch_size * total_targets.shape[1] * 8

    peak_bytes = weights_mem + grad_mem + micro_input_mem + micro_target_mem + activation_mem
    return int(peak_bytes)
