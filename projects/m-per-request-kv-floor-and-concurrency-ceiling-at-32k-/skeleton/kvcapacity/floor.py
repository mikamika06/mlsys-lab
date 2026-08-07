def get_dtype_bytes(dtype: str) -> float:
    raise NotImplementedError

def per_request_kv_bytes(model_config: dict, seq_len: int, kv_dtype: str = "float16") -> int:
    raise NotImplementedError

def model_weights_bytes(model_config: dict, model_dtype: str = "float16") -> int:
    raise NotImplementedError
