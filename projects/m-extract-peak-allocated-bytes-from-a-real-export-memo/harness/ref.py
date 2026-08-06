import random

def generate_timeline(seed=42):
    rng = random.Random(seed)
    events = []
    current = 0
    for _ in range(50):
        size = rng.randint(10, 1000) * 1024
        if current < 500000 or rng.random() > 0.4:
            events.append({"action": "alloc", "size": size})
            current += size
        else:
            events.append({"action": "free", "size": size})
            current = max(0, current - size)
    return {"events": events}

def extract_peak_allocated_bytes(timeline_data):
    peak = 0
    current = 0
    for ev in timeline_data.get("events", []):
        size = ev.get("size", 0)
        if ev.get("action") == "alloc":
            current += size
            if current > peak:
                peak = current
        elif ev.get("action") == "free":
            current -= size
    return peak

def generate_oom_snapshot(seed=42):
    rng = random.Random(seed)
    allocations = []
    for i in range(20):
        allocations.append({
            "id": i,
            "size": rng.randint(100, 5000) * 1024,
            "status": "live" if rng.random() > 0.3 else "dead"
        })
    return {"allocations": allocations}

def find_largest_live_allocation(oom_snapshot):
    allocations = oom_snapshot.get("allocations", [])
    largest = None
    max_size = -1
    for alloc in allocations:
        if alloc.get("status") == "live":
            if alloc.get("size", 0) > max_size:
                max_size = alloc.get("size", 0)
                largest = alloc
    return largest

def generate_fragmentation_workload(seed=42):
    rng = random.Random(seed)
    total_memory = 10485760
    ops = []
    for _ in range(30):
        op_type = "alloc" if rng.random() > 0.4 else "free"
        if op_type == "alloc":
            ops.append({"type": "alloc", "size": rng.randint(100, 2000) * 1024})
        else:
            ops.append({"type": "free", "block_index": rng.randint(0, 5)})
    return {"total_memory": total_memory, "ops": ops}

def simulate_allocator_fragmentation(operations):
    blocks = [{"size": operations.get("total_memory", 1024), "free": True}]
    peak_frag = 0.0
    for op in operations.get("ops", []):
        if op["type"] == "alloc":
            needed = op["size"]
            for i, b in enumerate(blocks):
                if b["free"] and b["size"] >= needed:
                    rem = b["size"] - needed
                    b["size"] = needed
                    b["free"] = False
                    if rem > 0:
                        blocks.insert(i + 1, {"size": rem, "free": True})
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
