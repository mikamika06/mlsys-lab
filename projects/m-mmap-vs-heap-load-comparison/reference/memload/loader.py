import math


def compare_load_footprint(tensors, page_size=4096):
    """Compare heap vs mmap loading memory footprint."""
    if not tensors:
        return {
            "heap_peak_bytes": 0,
            "heap_resident_bytes": 0,
            "mmap_virtual_bytes": 0,
            "mmap_resident_bytes": 0,
            "rss_savings_ratio": 0.0,
        }

    heap_resident_bytes = sum(t["size_bytes"] for t in tensors)
    max_tensor_size = max((t["size_bytes"] for t in tensors), default=0)
    heap_peak_bytes = heap_resident_bytes + max_tensor_size

    max_end_offset = max((t["offset"] + t["size_bytes"] for t in tensors), default=0)
    mmap_virtual_bytes = math.ceil(max_end_offset / page_size) * page_size if max_end_offset > 0 else 0

    touched_pages = set()
    for t in tensors:
        accessed = t.get("accessed_bytes", t["size_bytes"])
        if accessed <= 0:
            continue
        start_offset = t["offset"]
        end_offset = t["offset"] + accessed - 1
        start_page = start_offset // page_size
        end_page = end_offset // page_size
        for p in range(start_page, end_page + 1):
            touched_pages.add(p)

    mmap_resident_bytes = len(touched_pages) * page_size

    if heap_resident_bytes > 0:
        rss_savings_ratio = float(heap_resident_bytes - mmap_resident_bytes) / float(heap_resident_bytes)
    else:
        rss_savings_ratio = 0.0

    return {
        "heap_peak_bytes": heap_peak_bytes,
        "heap_resident_bytes": heap_resident_bytes,
        "mmap_virtual_bytes": mmap_virtual_bytes,
        "mmap_resident_bytes": mmap_resident_bytes,
        "rss_savings_ratio": rss_savings_ratio,
    }
