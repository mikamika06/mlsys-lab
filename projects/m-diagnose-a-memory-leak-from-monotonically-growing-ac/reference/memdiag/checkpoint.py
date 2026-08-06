def measure_checkpoint_reduction(layers, batch_size):
    if not layers:
        return 0
    normal_peak = sum(layers) * batch_size
    checkpointed_peak = max(layers) * batch_size + (sum(layers) // 2)
    return normal_peak - checkpointed_peak
