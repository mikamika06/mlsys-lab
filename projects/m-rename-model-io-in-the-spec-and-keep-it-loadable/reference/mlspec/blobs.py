import hashlib


def detect_duplicate_blobs(blobs):
    seen = {}
    duplicates = set()
    for path, data in sorted(blobs.items()):
        h = hashlib.sha256(data).hexdigest()
        if h in seen:
            duplicates.add(path)
            duplicates.add(seen[h])
        else:
            seen[h] = path
    return sorted(list(duplicates))
