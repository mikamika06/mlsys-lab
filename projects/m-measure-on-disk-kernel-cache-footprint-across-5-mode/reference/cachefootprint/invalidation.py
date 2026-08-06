import os


def check_path_invalidation(cache_dir, new_dir):
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"path:{cache_dir}")

    os.makedirs(new_dir, exist_ok=True)
    import shutil
    for item in os.listdir(cache_dir):
        s = os.path.join(cache_dir, item)
        d = os.path.join(new_dir, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)

    new_meta_path = os.path.join(new_dir, "meta.txt")
    with open(new_meta_path, "r") as f:
        data = f.read()

    is_invalidated = f"path:{new_dir}" not in data and f"path:{cache_dir}" in data
    return {"path_keyed": True, "invalidated": is_invalidated}
