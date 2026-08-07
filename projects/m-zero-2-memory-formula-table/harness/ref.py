import numpy as np
from zerotwo.memory import zero2_memory_breakdown as ref_zero2_memory
from zerotwo.comm import calc_bucket_count as ref_calc_bucket_count, toy_reduce_scatter as ref_toy_reduce_scatter

TEST_CONFIGS = [
    {"num_params": 1000000, "world_size": 4, "bytes_per_param": 2, "optimizer_bytes_per_param": 12, "activation_bytes": 500000},
    {"num_params": 70000000, "world_size": 8, "bytes_per_param": 2, "optimizer_bytes_per_param": 16, "activation_bytes": 10000000},
    {"num_params": 500000000, "world_size": 16, "bytes_per_param": 4, "optimizer_bytes_per_param": 12, "activation_bytes": 0},
    {"num_params": 1500000000, "world_size": 32, "bytes_per_param": 2, "optimizer_bytes_per_param": 12, "activation_bytes": 2000000}
]

BUCKET_TESTS = [
    {"total_elements": 1048576, "element_size": 2, "allgather_bucket_size_bytes": 524288},
    {"total_elements": 5000000, "element_size": 4, "allgather_bucket_size_bytes": 1048576},
    {"total_elements": 1000, "element_size": 2, "allgather_bucket_size_bytes": 10000}
]

REDUCE_SCATTER_TESTS = [
    {"grads": [np.arange(16, dtype=np.float32), np.arange(16, dtype=np.float32) * 2], "world_size": 2},
    {"grads": [np.ones(32, dtype=np.float32), np.zeros(32, dtype=np.float32), np.ones(32, dtype=np.float32) * 3, np.ones(32, dtype=np.float32)], "world_size": 4}
]
