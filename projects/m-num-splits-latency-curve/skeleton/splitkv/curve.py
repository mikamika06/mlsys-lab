def predict_latency(batch_size: int, seq_len: int, num_splits: int, num_sm: int = 108) -> float:
    raise NotImplementedError


def optimal_num_splits(batch_size: int, seq_len: int, num_sm: int = 108) -> int:
    raise NotImplementedError
