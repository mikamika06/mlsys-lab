import os

def enumerate_debug_dir(dir_path: str) -> list:
    if not os.path.exists(dir_path):
        return []
    items = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            items.append(os.path.relpath(os.path.join(root, f), dir_path))
    return sorted(items)
