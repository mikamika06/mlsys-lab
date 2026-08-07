import numpy as np


def merge_tp_weights(shards, tp_size, mode):
    result = {}
    for base_key in sorted(list(set(k.rsplit("_rank", 1)[0] for k in shards.keys()))):
        rank_tensors = [shards[f"{base_key}_rank{i}"] for i in range(tp_size)]
        if mode == "column":
            result[base_key] = np.concatenate(rank_tensors, axis=0)
        elif mode == "row":
            result[base_key] = np.concatenate(rank_tensors, axis=1)
        else:
            raise ValueError(f"unknown mode {mode}")
    return result
