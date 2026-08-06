import numpy as np


CONFIGS = [
    {"base": 10000.0, "seq_len": 4096, "max_pos": 2048, "orig_max": 2048, "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "factor": 2.0, "low_freq": 1.0, "high_freq": 4.0},
    {"base": 10000.0, "seq_len": 8192, "max_pos": 4096, "orig_max": 4096, "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "factor": 2.0, "low_freq": 1.0, "high_freq": 4.0},
    {"base": 500000.0, "seq_len": 16384, "max_pos": 8192, "orig_max": 8192, "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "factor": 4.0, "low_freq": 1.0, "high_freq": 4.0},
]


def compute_dynamic_ntk_base(base, seq_len, max_position_embeddings):
    if seq_len <= max_position_embeddings:
        return float(base)
    return float(base * (seq_len / max_position_embeddings) ** (2.0 / 2.0))


def compute_yarn_parameters(base, seq_len, max_position_embeddings, original_max_position_embeddings, beta_fast, beta_slow, mscale):
    scale = max(1.0, seq_len / original_max_position_embeddings)
    dim = 128
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    if scale <= 1.0:
        return inv_freq, 1.0
    low = original_max_position_embeddings / beta_fast
    high = original_max_position_embeddings / beta_slow

    def ramp(min_val, max_val, val):
        if val <= min_val:
            return 0.0
        if val >= max_val:
            return 1.0
        return (val - min_val) / (max_val - min_val)

    inv_freq_s = []
    for freq in inv_freq:
        wavelength = 2 * np.pi / freq
        factor = ramp(low, high, wavelength)
        inv_freq_s.append(freq / scale * (1 - factor) + freq * factor)

    mscale_val = 0.1 * np.log(scale) + 1.0
    return np.array(inv_freq_s, dtype=np.float32), float(mscale_val)


def compute_llama3_scaling(base, seq_len, max_position_embeddings, original_max_position_embeddings, factor, low_freq_factor, high_freq_factor):
    dim = 128
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    if seq_len <= original_max_position_embeddings:
        return inv_freq
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor

    inv_freq_llama = []
    for freq in inv_freq:
        wavelength = 2 * np.pi / freq
        if wavelength < high_freq_wavelen:
            inv_freq_llama.append(freq)
        elif wavelength > low_freq_wavelen:
            inv_freq_llama.append(freq / factor)
        else:
            smooth = (original_max_position_embeddings / wavelength - high_freq_factor) / (low_freq_factor - high_freq_factor)
            interpolated = freq / factor
            computed = (1 - smooth) * interpolated + smooth * freq
            inv_freq_llama.append(computed)
    return np.array(inv_freq_llama, dtype=np.float32)
