def build_blob_index(config):
    by_digest = {}
    for tag, blobs in config["tags"].items():
        for b in blobs:
            entry = by_digest.setdefault(
                b["digest"], {"digest": b["digest"], "size": b["size"], "tags": set()}
            )
            entry["tags"].add(tag)
    return [
        {"digest": d, "size": by_digest[d]["size"], "tags": sorted(by_digest[d]["tags"])}
        for d in sorted(by_digest)
    ]
