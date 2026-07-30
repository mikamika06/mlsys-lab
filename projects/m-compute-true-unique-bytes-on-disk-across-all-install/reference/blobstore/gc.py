def find_orphaned_blobs(config, disk_blobs):
    referenced = {b["digest"] for blobs in config["tags"].values() for b in blobs}
    return sorted(d for d in disk_blobs if d not in referenced)


def orphaned_bytes(config, disk_blobs):
    return sum(disk_blobs[d] for d in find_orphaned_blobs(config, disk_blobs))
