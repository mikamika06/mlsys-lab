import os


def inspect_cache(cache_dir):
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
