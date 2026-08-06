def parse_log(text):
    """Parse TRT EP log text to extract subgraphs and cache state."""
    subgraphs = []
    cache_reused = False
    for line in text.splitlines():
        if "Subgraph" in line:
            parts = line.split(":")
            sub_id = parts[0].strip()
            nodes = int(parts[1].strip().split()[0])
            subgraphs.append({"id": sub_id, "nodes": nodes})
        if "reused" in line.lower() or "loaded" in line.lower():
            cache_reused = True
    return {"subgraphs": subgraphs, "count": len(subgraphs), "cache_reused": cache_reused}
