def compute_kv_bytes(config: dict, prompt_len: int) -> int:
    raise NotImplementedError

def compute_transfer_times(kv_bytes: int, bandwidths_gbps: list) -> dict:
    raise NotImplementedError
