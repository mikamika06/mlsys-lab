from memload.loader import compare_load_footprint


def calculate_dedup_savings(tensors, page_size=4096):
    """Calculate deduplication savings for disk, heap, and mmap."""
    if not tensors:
        return {
            "raw_total_bytes": 0,
            "unique_total_bytes": 0,
            "disk_savings_bytes": 0,
            "heap_savings_bytes": 0,
            "mmap_savings_bytes": 0,
            "heap_dedup_savings_ratio": 0.0,
            "mmap_dedup_savings_ratio": 0.0,
        }

    raw_total_bytes = sum(t["size_bytes"] for t in tensors)
    raw_load = compare_load_footprint(tensors, page_size=page_size)
    raw_mmap_resident = raw_load["mmap_resident_bytes"]

    seen_hashes = {}
    for t in tensors:
        h = t["hash"]
        if h not in seen_hashes:
            seen_hashes[h] = {
                "name": t["name"],
                "hash": h,
                "size_bytes": t["size_bytes"],
                "accessed_bytes": t.get("accessed_bytes", t["size_bytes"]),
            }
        else:
            seen_hashes[h]["accessed_bytes"] = max(
                seen_hashes[h]["accessed_bytes"],
                t.get("accessed_bytes", t["size_bytes"]),
            )

    unique_tensors = []
    current_offset = 0
    for h, ut in seen_hashes.items():
        unique_tensors.append({
            "name": ut["name"],
            "hash": ut["hash"],
            "offset": current_offset,
            "size_bytes": ut["size_bytes"],
            "accessed_bytes": ut["accessed_bytes"],
        })
        current_offset += ut["size_bytes"]

    unique_total_bytes = sum(ut["size_bytes"] for ut in unique_tensors)
    disk_savings_bytes = raw_total_bytes - unique_total_bytes
    heap_savings_bytes = raw_total_bytes - unique_total_bytes

    heap_savings_ratio = (
        float(heap_savings_bytes) / float(raw_total_bytes)
        if raw_total_bytes > 0
        else 0.0
    )

    dedup_load = compare_load_footprint(unique_tensors, page_size=page_size)
    dedup_mmap_resident = dedup_load["mmap_resident_bytes"]
    mmap_savings_bytes = raw_mmap_resident - dedup_mmap_resident

    mmap_savings_ratio = (
        float(mmap_savings_bytes) / float(raw_mmap_resident)
        if raw_mmap_resident > 0
        else 0.0
    )

    return {
        "raw_total_bytes": raw_total_bytes,
        "unique_total_bytes": unique_total_bytes,
        "disk_savings_bytes": disk_savings_bytes,
        "heap_savings_bytes": heap_savings_bytes,
        "mmap_savings_bytes": mmap_savings_bytes,
        "heap_dedup_savings_ratio": heap_savings_ratio,
        "mmap_dedup_savings_ratio": mmap_savings_ratio,
    }
