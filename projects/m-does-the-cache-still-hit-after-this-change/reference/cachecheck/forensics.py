import os


def analyze_cache_dir(dir_path):
    files = []
    if not os.path.exists(dir_path):
        return {"total_files": 0, "total_bytes": 0, "entries": []}
    total_bytes = 0
    for root, _, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(root, f)
            size = os.path.getsize(fp)
            total_bytes += size
            files.append({"name": f, "path": fp, "size": size})
    files.sort(key=lambda x: x["name"])
    return {"total_files": len(files), "total_bytes": total_bytes, "entries": files}
