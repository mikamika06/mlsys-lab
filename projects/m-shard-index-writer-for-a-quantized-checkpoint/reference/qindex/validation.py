import json


def validate_index_structure(serialized_json):
    try:
        data = json.loads(serialized_json)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if "weight_map" not in data or "metadata" not in data:
        return False
    for tname, info in data["weight_map"].items():
        if not all(k in info for k in ("file", "dtype", "shape", "offsets")):
            return False
        if not isinstance(info["offsets"], list) or len(info["offsets"]) != 2:
            return False
        if info["offsets"][0] > info["offsets"][1]:
            return False
    return True
