import math


def plan_activation_arena(buffers: list[dict], default_alignment: int = 64) -> dict:
    if not buffers:
        return {"arena_size": 0, "offsets": {}}

    def sort_key(b):
        return (-b["size"], b["liveness"][0], str(b["id"]))

    sorted_bufs = sorted(buffers, key=sort_key)
    placed = []

    for buf in sorted_bufs:
        buf_id = buf["id"]
        size = buf["size"]
        start, end = buf["liveness"]
        align = buf.get("alignment", default_alignment)

        cand_offset = 0
        while True:
            conflict = False
            for p in placed:
                p_start, p_end = p["liveness"]
                if max(start, p_start) <= min(end, p_end):
                    p_off = p["offset"]
                    p_size = p["size"]
                    if not (cand_offset + size <= p_off or p_off + p_size <= cand_offset):
                        conflict = True
                        break
            if not conflict:
                break
            cand_offset += align

        placed.append({
            "id": buf_id,
            "offset": cand_offset,
            "size": size,
            "liveness": (start, end),
        })

    max_end = max(p["offset"] + p["size"] for p in placed) if placed else 0
    arena_size = math.ceil(max_end / default_alignment) * default_alignment if max_end > 0 else 0
    offsets = {p["id"]: p["offset"] for p in placed}
    return {"arena_size": arena_size, "offsets": offsets}
