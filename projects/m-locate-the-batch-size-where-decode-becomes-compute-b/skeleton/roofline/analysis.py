def find_decode_compute_bound_batch_size(model_config, hardware_specs):
    raise NotImplementedError


def calculate_operational_intensity(model_config, batch_size, seq_len, phase="decode"):
    raise NotImplementedError
