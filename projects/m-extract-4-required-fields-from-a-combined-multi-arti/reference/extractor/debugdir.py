import os

def enumerate_debug_directory(dir_path):
    if not os.path.isdir(dir_path):
        return []
    files = []
    for root, _, filenames in os.walk(dir_path):
        for f in filenames:
            files.append(f)
    return sorted(files)
