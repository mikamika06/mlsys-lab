from kvbytes.calc import calc_bytes_per_token


def measure_growth(config, num_tokens, dtype_bytes=2):
    per_token = calc_bytes_per_token(config, dtype_bytes)
    return [t * per_token for t in range(1, num_tokens + 1)]
