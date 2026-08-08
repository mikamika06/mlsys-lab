import math


def compare_load_footprint(tensors, page_size=4096):
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
    rss_savings_ratio = (
        float(heap_resident_bytes - mmap_resident_bytes) / float(heap_resident_bytes)
        if heap_resident_bytes > 0
        else 0.0
    )
    return {
        "heap_peak_bytes": heap_peak_bytes,
        "heap_resident_bytes": heap_resident_bytes,
        "mmap_virtual_bytes": mmap_virtual_bytes,
        "mmap_resident_bytes": mmap_resident_bytes,
        "rss_savings_ratio": rss_savings_ratio,
    }


def attribute_size_regression(base_tensors, candidate_tensors):
    base_map = {t["name"]: t for t in base_tensors}
    cand_map = {t["name"]: t for t in candidate_tensors}
    base_total = sum(t["size_bytes"] for t in base_tensors)
    cand_total = sum(t["size_bytes"] for t in candidate_tensors)
    net_delta = cand_total - base_total
    all_layers = set()
    for t in base_tensors:
        all_layers.add(t.get("layer", "default"))
    for t in candidate_tensors:
        all_layers.add(t.get("layer", "default"))
    by_layer = {layer: 0 for layer in sorted(all_layers)}
    for t in candidate_tensors:
        by_layer[t.get("layer", "default")] += t["size_bytes"]
    for t in base_tensors:
        by_layer[t.get("layer", "default")] -= t["size_bytes"]
    added_delta = 0
    removed_delta = 0
    modified_delta = 0
    all_names = set(base_map.keys()) | set(cand_map.keys())
    contributors = []
    for name in all_names:
        in_base = name in base_map
        in_cand = name in cand_map
        if in_cand and not in_base:
            c_t = cand_map[name]
            delta = c_t["size_bytes"]
            added_delta += delta
            contributors.append({"name": name, "layer": c_t.get("layer", "default"), "delta_bytes": delta})
        elif in_base and not in_cand:
            b_t = base_map[name]
            delta = -b_t["size_bytes"]
            removed_delta += delta
            contributors.append({"name": name, "layer": b_t.get("layer", "default"), "delta_bytes": delta})
        else:
            b_t = base_map[name]
            c_t = cand_map[name]
            delta = c_t["size_bytes"] - b_t["size_bytes"]
            if delta != 0:
                modified_delta += delta
                contributors.append({"name": name, "layer": c_t.get("layer", "default"), "delta_bytes": delta})
    contributors.sort(key=lambda x: (-abs(x["delta_bytes"]), x["name"]))
    return {
        "total_base_bytes": base_total,
        "total_candidate_bytes": cand_total,
        "net_delta_bytes": net_delta,
        "by_layer": by_layer,
        "category_deltas": {
            "added": added_delta,
            "removed": removed_delta,
            "modified": modified_delta,
        },
        "top_contributors": contributors,
    }


def calculate_dedup_savings(tensors, page_size=4096):
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
        float(heap_savings_bytes) / float(raw_total_bytes) if raw_total_bytes > 0 else 0.0
    )
    dedup_load = compare_load_footprint(unique_tensors, page_size=page_size)
    dedup_mmap_resident = dedup_load["mmap_resident_bytes"]
    mmap_savings_bytes = raw_mmap_resident - dedup_mmap_resident
    mmap_savings_ratio = (
        float(mmap_savings_bytes) / float(raw_mmap_resident) if raw_mmap_resident > 0 else 0.0
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


def _gen_load_cases():
    cases = []
    t1 = []
    offset = 0
    for i in range(10):
        sz = (i + 1) * 2000
        t1.append({"name": f"t_{i}", "offset": offset, "size_bytes": sz, "accessed_bytes": sz})
        offset += sz
    cases.append({"tensors": t1, "page_size": 4096})

    t2 = []
    offset = 0
    for i in range(8):
        sz = 16384
        acc = 1000 if i % 2 == 0 else 16384
        t2.append({"name": f"t_{i}", "offset": offset, "size_bytes": sz, "accessed_bytes": acc})
        offset += sz + 8192
    cases.append({"tensors": t2, "page_size": 4096})

    t3 = []
    offset = 0
    for i in range(12):
        sz = 8000 + i * 500
        acc = 4000
        t3.append({"name": f"t_{i}", "offset": offset, "size_bytes": sz, "accessed_bytes": acc})
        offset += sz
    cases.append({"tensors": t3, "page_size": 65536})

    t4 = []
    offset = 0
    for i in range(6):
        sz = 100000
        acc = 0 if i >= 3 else sz
        t4.append({"name": f"t_{i}", "offset": offset, "size_bytes": sz, "accessed_bytes": acc})
        offset += sz
    cases.append({"tensors": t4, "page_size": 4096})

    cases.append({"tensors": [], "page_size": 4096})
    return cases


def _gen_attribution_cases():
    cases = []
    b1 = [
        {"name": "l0.w", "layer": "layer_0", "size_bytes": 1000},
        {"name": "l0.b", "layer": "layer_0", "size_bytes": 100},
        {"name": "l1.w", "layer": "layer_1", "size_bytes": 2000},
        {"name": "l2.w", "layer": "layer_2", "size_bytes": 5000},
    ]
    c1 = [
        {"name": "l0.w", "layer": "layer_0", "size_bytes": 1200},
        {"name": "l0.b", "layer": "layer_0", "size_bytes": 100},
        {"name": "l2.w", "layer": "layer_2", "size_bytes": 5000},
        {"name": "l3.w", "layer": "layer_3", "size_bytes": 3000},
    ]
    cases.append({"base": b1, "candidate": c1})

    b2 = [{"name": f"w_{i}", "layer": f"layer_{i//2}", "size_bytes": 1000 * (i + 1)} for i in range(6)]
    c2 = [{"name": f"w_{i}", "layer": f"layer_{i//2}", "size_bytes": 1500 * (i + 1)} for i in range(6)]
    cases.append({"base": b2, "candidate": c2})

    b3 = [{"name": f"w_{i}", "layer": f"layer_{i}", "size_bytes": 4096} for i in range(10)]
    c3 = [{"name": f"w_{i}", "layer": f"layer_{i}", "size_bytes": 4096} for i in range(5)]
    cases.append({"base": b3, "candidate": c3})

    cases.append({"base": b1, "candidate": b1})
    cases.append({"base": b3, "candidate": c1})
    return cases


def _gen_dedup_cases():
    cases = []
    t1 = []
    offset = 0
    hashes = ["h_embed", "h_attn", "h_attn", "h_mlp", "h_attn", "h_embed"]
    sizes = [65536, 131072, 131072, 262144, 131072, 65536]
    accessed = [65536, 131072, 65536, 262144, 131072, 32768]
    for i, (h, sz, acc) in enumerate(zip(hashes, sizes, accessed)):
        t1.append({"name": f"tensor_{i}", "hash": h, "offset": offset, "size_bytes": sz, "accessed_bytes": acc})
        offset += sz
    cases.append({"tensors": t1, "page_size": 4096})

    t2 = []
    offset = 0
    for i in range(5):
        sz = 16384 * (i + 1)
        t2.append({"name": f"unique_{i}", "hash": f"hash_{i}", "offset": offset, "size_bytes": sz, "accessed_bytes": sz})
        offset += sz
    cases.append({"tensors": t2, "page_size": 4096})

    t3 = []
    offset = 0
    for i in range(8):
        sz = 32768
        t3.append({"name": f"dup_{i}", "hash": "same_hash", "offset": offset, "size_bytes": sz, "accessed_bytes": sz})
        offset += sz
    cases.append({"tensors": t3, "page_size": 4096})

    t4 = []
    offset = 0
    for i in range(6):
        h = f"hash_{i % 2}"
        sz = 50000
        acc = 10000 if i % 2 == 0 else 50000
        t4.append({"name": f"t_{i}", "hash": h, "offset": offset, "size_bytes": sz, "accessed_bytes": acc})
        offset += sz
    cases.append({"tensors": t4, "page_size": 65536})

    cases.append({"tensors": [], "page_size": 4096})
    return cases


LOAD_CASES = _gen_load_cases()
ATTRIBUTION_CASES = _gen_attribution_cases()
DEDUP_CASES = _gen_dedup_cases()
