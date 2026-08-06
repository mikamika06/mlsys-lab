def compute_accumulation_steps(target_effective_batch_size, per_device_batch_size, num_devices):
    current_batch = per_device_batch_size * num_devices
    if target_effective_batch_size % current_batch != 0:
        raise ValueError("Target effective batch size must be divisible by per-device batch size times num devices.")
    return target_effective_batch_size // current_batch
