import hashlib
import json
import os


def compute_stable_hash(payload):
    def sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in sorted(obj.items())
                    if not str(k).startswith("_volatile")}
        if isinstance(obj, (list, tuple)):
            return [sanitize(x) for x in obj]
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    cleaned = sanitize(payload)
    raw = json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def analyze_cache_directory(cache_dir):
    files = []
    total_size = 0
    if not os.path.exists(cache_dir):
        return {"file_count": 0, "total_size_bytes": 0, "largest_file": None}
    for root, _, filenames in os.walk(cache_dir):
        for f in filenames:
            p = os.path.join(root, f)
            sz = os.path.getsize(p)
            files.append((p, sz))
            total_size += sz
    if not files:
        return {"file_count": 0, "total_size_bytes": 0, "largest_file": None}
    files.sort(key=lambda x: x[1], reverse=True)
    largest = os.path.relpath(files[0][0], cache_dir)
    return {
        "file_count": len(files),
        "total_size_bytes": total_size,
        "largest_file": largest
    }


PAYLOADS = [
    {"model": "llama", "lr": 0.001, "layers": [1, 2, 3], "_volatile_timestamp": 12345},
    {"model": "llama", "lr": 0.0010001, "layers": [1, 2, 3], "_volatile_id": "abc"},
    {"layers": [3, 2, 1], "model": "llama", "lr": 0.001, "_volatile_pid": 999},
    {"model": "mistral", "lr": 0.0005, "layers": [4, 5], "_volatile_debug": True},
    {"model": "opt", "lr": 0.01, "layers": [10], "_volatile_session": "xyz"}
]
