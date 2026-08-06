import os


def compare_bytes(cache_dir):
    raw_bytes = 0
    artifact_bytes = 0
    for root, _, files in os.walk(cache_dir):
        for file in files:
            path = os.path.join(root, file)
            size = os.path.getsize(path)
            raw_bytes += size
            if file.endswith(".so") or file.endswith(".bin"):
                artifact_bytes += size
    return {"raw_bytes": raw_bytes, "artifact_bytes": artifact_bytes}
