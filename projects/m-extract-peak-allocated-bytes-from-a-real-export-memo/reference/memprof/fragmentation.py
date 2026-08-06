def simulate_allocator_fragmentation(operations):
    blocks = [{"size": operations.get("total_memory", 1024), "free": True}]
    peak_frag = 0.0
    for op in operations.get("ops", []):
        if op["type"] == "alloc":
            needed = op["size"]
            allocated = False
            for i, b in enumerate(blocks):
                if b["free"] and b["size"] >= needed:
                    rem = b["size"] - needed
                    b["size"] = needed
                    b["free"] = False
                    if rem > 0:
                        blocks.insert(i + 1, {"size": rem, "free": True})
                    allocated = True
                    break
        elif op["type"] == "free":
            idx = op["block_index"]
            if 0 <= idx < len(blocks):
                blocks[idx]["free"] = True
                new_blocks = []
                i = 0
                while i < len(blocks):
                    curr = blocks[i]
                    while i + 1 < len(blocks) and curr["free"] and blocks[i + 1]["free"]:
                        curr["size"] += blocks[i + 1]["size"]
                        i += 1
                    new_blocks.append(curr)
                    i += 1
                blocks = new_blocks

        total_free = sum(b["size"] for b in blocks if b["free"])
        max_free_block = max((b["size"] for b in blocks if b["free"]), default=0)
        if total_free > 0:
            frag = 1.0 - (max_free_block / total_free)
            if frag > peak_frag:
                peak_frag = frag
    return peak_frag
