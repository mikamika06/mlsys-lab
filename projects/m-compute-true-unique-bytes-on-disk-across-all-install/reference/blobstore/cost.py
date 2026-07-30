from . import index


def unique_bytes_on_disk(config):
    return sum(b["size"] for b in index.build_blob_index(config))


def naive_total_bytes(config):
    return sum(b["size"] for blobs in config["tags"].values() for b in blobs)


def incremental_pull_bytes(config, candidate):
    have = {b["digest"] for blobs in config["tags"].values() for b in blobs}
    seen = set()
    total = 0
    for b in candidate:
        d = b["digest"]
        if d in have or d in seen:
            continue
        seen.add(d)
        total += b["size"]
    return total
