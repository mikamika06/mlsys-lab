"""Memory arena growth and peak RSS analyzer implementation."""


def analyze_arena_vs_rss(allocations, deallocations, block_size):
    active_bytes = 0
    arena_reserved_bytes = 0
    peak_active = 0
    peak_arena = 0
    timeline = []

    events = []
    for a in allocations:
        events.append((a["time"], "alloc", a["id"], a["size"]))
    for d in deallocations:
        events.append((d["time"], "dealloc", d["id"], d["size"]))
    events.sort(key=lambda x: (x[0], 0 if x[1] == "dealloc" else 1))

    for time, ev_type, alloc_id, size in events:
        if ev_type == "alloc":
            active_bytes += size
            if active_bytes > arena_reserved_bytes:
                needed = active_bytes - arena_reserved_bytes
                blocks = (needed + block_size - 1) // block_size
                arena_reserved_bytes += blocks * block_size
        else:
            active_bytes -= size

        peak_active = max(peak_active, active_bytes)
        peak_arena = max(peak_arena, arena_reserved_bytes)
        timeline.append({
            "time": time,
            "active_bytes": active_bytes,
            "arena_reserved_bytes": arena_reserved_bytes,
        })

    waste_at_peak = peak_arena - peak_active
    efficiency = peak_active / peak_arena if peak_arena > 0 else 1.0

    return {
        "peak_active_bytes": peak_active,
        "peak_arena_bytes": peak_arena,
        "waste_at_peak_bytes": waste_at_peak,
        "efficiency": efficiency,
        "timeline": timeline,
    }
