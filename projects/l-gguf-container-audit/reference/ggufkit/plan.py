from . import container


def _page_span(start, end, page):
    first = start // page
    last = (end - 1) // page
    return first, last


def load_plan(blob, page=16384, want=None):
    index = container.parse_tensor_index(blob)
    size = len(blob)
    tensors = index["tensors"]
    if want is not None:
        want = set(want)
        chosen = [t for t in tensors if t["name"] in want]
        missing = sorted(want - {t["name"] for t in tensors})
        if missing:
            raise KeyError("no such tensor: %s" % ", ".join(missing))
    else:
        chosen = list(tensors)

    weight_bytes = sum(t["n_bytes"] for t in tensors)
    pages = set()
    per_tensor = []
    for t in chosen:
        start = t["absolute_data_offset"]
        end = start + t["n_bytes"]
        first, last = _page_span(start, end, page)
        for p in range(first, last + 1):
            pages.add(p)
        per_tensor.append({
            "name": t["name"],
            "byte_range": [start, end],
            "page_range": [first, last],
            "pages": last - first + 1,
            "resident_bytes": (last - first + 1) * page,
            "waste_bytes": (last - first + 1) * page - (end - start),
        })

    header_bytes = index["data_start"]
    resident = len(pages) * page
    return {
        "page_size": page,
        "file_bytes": size,
        "metadata_bytes": header_bytes,
        "weight_bytes": weight_bytes,
        "metadata_fraction": header_bytes / size if size else 0.0,
        "selected_tensors": len(chosen),
        "distinct_pages": len(pages),
        "resident_bytes": resident,
        "resident_fraction": resident / size if size else 0.0,
        "shared_pages": sum(t["pages"] for t in per_tensor) - len(pages),
        "tensors": per_tensor,
    }


def alignment_report(blob, page=16384):
    index = container.parse_tensor_index(blob)
    rows = []
    for t in index["tensors"]:
        start = t["absolute_data_offset"]
        rows.append({
            "name": t["name"],
            "offset": start,
            "aligned_to_container": start % index["alignment"] == 0,
            "aligned_to_page": start % page == 0,
            "page_offset": start % page,
        })
    return {"alignment": index["alignment"], "page_size": page, "tensors": rows}
