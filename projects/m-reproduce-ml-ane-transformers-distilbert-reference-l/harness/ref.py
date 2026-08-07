import numpy as np


def get_reference_latencies():
    return {
        "embedding": 1.2,
        "attention": 4.5,
        "ffn": 3.1,
        "layer_norm": 0.5,
    }


def compute_latency_ratio(latencies):
    ordered_keys = sorted(latencies.keys(), key=lambda k: latencies[k])
    ref_keys = sorted(get_reference_latencies().keys(), key=lambda k: get_reference_latencies()[k])
    return 1.0 if ordered_keys == ref_keys else 0.0


def standard_softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def split_softmax(x, chunks=2):
    sub_x = np.array_split(x, chunks, axis=-1)
    maxs = [np.max(s, axis=-1, keepdims=True) for s in sub_x]
    global_max = np.maximum.reduce(maxs)

    numerators = []
    denominators = []
    for s, m in zip(sub_x, maxs):
        num = np.exp(s - global_max)
        numerators.append(num)
        denominators.append(np.sum(num, axis=-1, keepdims=True))

    total_denom = sum(denominators)
    return [num / total_denom for num in numerators]


def derive_l2_chunk_size(tensor_shape, element_size_bytes, l2_cache_capacity):
    total_elements = np.prod(tensor_shape[:-1])
    bytes_per_row = tensor_shape[-1] * element_size_bytes
    if bytes_per_row >= l2_cache_capacity:
        return max(1, l2_cache_capacity // element_size_bytes)
    target_bytes = l2_cache_capacity // 2
    chunk_rows = max(1, target_bytes // bytes_per_row)
    return int(min(chunk_rows, tensor_shape[0]))
