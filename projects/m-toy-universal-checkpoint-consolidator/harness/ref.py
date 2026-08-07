import numpy as np

def generate_index_fixtures():
    idx_good = {"weight_map": {"A": "1.bin", "B": "1.bin", "C": "2.bin"}}
    fs_good = {"1.bin": 100, "2.bin": 200}
    idx_bad = {"weight_map": {"A": "1.bin", "B": "3.bin"}}
    fs_bad = {"1.bin": 100}
    return [(idx_good, fs_good), (idx_bad, fs_bad)]

def generate_shards(tp_degree):
    np.random.seed(42)
    shards = []
    for i in range(tp_degree):
        s = {
            "w_col": np.ones((4, 8)) * i,
            "w_row": np.ones((8, 4)) * i,
            "w_rep": np.ones((8, 8))
        }
        shards.append(s)
    return shards, {"w_col": 0, "w_row": 1, "w_rep": None}

def verify_index(index_dict, file_sizes_dict):
    weight_map = index_dict.get("weight_map", {})
    referenced_files = set(weight_map.values())
    actual_files = set(file_sizes_dict.keys())
    missing_files = sorted(list(referenced_files - actual_files))
    return {
        "is_valid": len(missing_files) == 0,
        "missing_files": missing_files,
        "total_files_referenced": len(referenced_files)
    }

def merge_tp_shards(shards, axis_map):
    if not shards:
        return {}
    keys = shards[0].keys()
    out = {}
    for k in keys:
        axis = axis_map.get(k, None)
        if axis is None:
            out[k] = shards[0][k]
        else:
            out[k] = np.concatenate([s[k] for s in shards], axis=axis)
    return out
