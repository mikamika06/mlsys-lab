import numpy as np
import math

def expected_drop_rate(capacity_factor: float, num_experts: int, seq_len: int) -> float:
    raise NotImplementedError

def recommend_capacity_factor(num_experts: int, seq_len: int, target_drop_rate: float) -> float:
    raise NotImplementedError
