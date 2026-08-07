def calculate_weight_bytes(config):
    raise NotImplementedError


def calculate_kv_cache_bytes(config, seq_len, batch_size=1):
    raise NotImplementedError


def calculate_activation_bytes(config, seq_len, batch_size=1):
    raise NotImplementedError


def predict_resident_vram(config, seq_len, batch_size=1):
    raise NotImplementedError
