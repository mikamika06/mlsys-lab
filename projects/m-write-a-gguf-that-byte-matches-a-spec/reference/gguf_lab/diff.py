import struct

def _parse_summary(data):
    if len(data) < 16:
        return None
    version = struct.unpack("<I", data[4:8])[0]
    n_kv = struct.unpack("<Q", data[8:16])[0]
    ptr = 16
    kv = {}
    for _ in range(n_kv):
        if ptr + 8 > len(data):
            break
        k_len = struct.unpack("<Q", data[ptr:ptr+8])[0]
        ptr += 8
        k = data[ptr:ptr+k_len].decode("utf-8")
        ptr += k_len
        if ptr + 4 > len(data):
            break
        vtype = struct.unpack("<I", data[ptr:ptr+4])[0]
        ptr += 4
        if vtype == 5:
            if ptr + 8 > len(data):
                break
            v_len = struct.unpack("<Q", data[ptr:ptr+8])[0]
            ptr += 8
            vval = data[ptr:ptr+v_len].decode("utf-8")
            ptr += v_len
        elif vtype in (2, 3):
            vval = struct.unpack("<I", data[ptr:ptr+4])[0]
            ptr += 4
        elif vtype == 4:
            vval = struct.unpack("<f", data[ptr:ptr+4])[0]
            ptr += 4
        else:
            vval = struct.unpack("<Q", data[ptr:ptr+8])[0]
            ptr += 8
        kv[k] = (vtype, vval)
    return {"version": version, "kv": kv}

def diff_gguf(data_a, data_b):
    sum_a = _parse_summary(data_a)
    sum_b = _parse_summary(data_b)
    if not sum_a or not sum_b:
        return {"error": "invalid parsing"}
    diff = {
        "version_changed": sum_a["version"] != sum_b["version"],
        "version_a": sum_a["version"],
        "version_b": sum_b["version"],
        "kv_added": sorted(list(set(sum_b["kv"].keys()) - set(sum_a["kv"].keys()))),
        "kv_removed": sorted(list(set(sum_a["kv"].keys()) - set(sum_b["kv"].keys()))),
        "kv_changed": sorted([k for k in set(sum_a["kv"].keys()).intersection(set(sum_b["kv"].keys())) if sum_a["kv"][k] != sum_b["kv"][k]])
    }
    return diff
