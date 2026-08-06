import os


def attribute_bytes(package_dir):
    attribution = {}
    total = 0
    for root, _, files in os.walk(package_dir):
        for name in files:
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, package_dir)
            size = os.path.getsize(full_path)
            attribution[rel_path] = size
            total += size
    attribution["_total"] = total
    return attribution
