import struct

def diff_gguf(data_a, data_b):
    if data_a == data_b:
        return {"status": "identical"}
    diffs = []
    if len(data_a) != len(data_b):
        diffs.append(f"length mismatch: {len(data_a)} vs {len(data_b)}")
    min_len = min(len(data_a), len(data_b))
    for i in range(min_len):
        if data_a[i] != data_b[i]:
            diffs.append(f"byte mismatch at index {i}")
            break
    return {"status": "different", "diffs": diffs}
