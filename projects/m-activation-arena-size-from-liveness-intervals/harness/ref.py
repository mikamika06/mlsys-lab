import math
import struct


def make_pte_data(segments, version=1, magic=b"PTE1"):
    buf = struct.pack("<4sII", magic, version, len(segments))
    for seg_type, offset, size, align in segments:
        buf += struct.pack("<IQQI", seg_type, offset, size, align)
    return buf


def reference_parse_pte_constants(pte_bytes: bytes) -> dict:
    if len(pte_bytes) < 12:
        raise ValueError("Invalid PTE header size")
    magic, version, num_segments = struct.unpack("<4sII", pte_bytes[:12])
    if magic != b"PTE1" or version != 1:
        raise ValueError("Invalid magic or version")
    ptr = 12
    for _ in range(num_segments):
        if ptr + 24 > len(pte_bytes):
            raise ValueError("Truncated segment table")
        seg_type, offset, size, align = struct.unpack("<IQQI", pte_bytes[ptr : ptr + 24])
        ptr += 24
        if seg_type == 1:
            return {"offset": int(offset), "size": int(size), "alignment": int(align)}
    raise ValueError("Constant segment not found")


def reference_plan_activation_arena(buffers: list[dict], default_alignment: int = 64) -> dict:
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


PTE_TEST_CASES = [
    {
        "segments": [(0, 0, 1024, 64), (1, 4096, 2048, 64), (2, 8192, 1024, 32)],
        "version": 1,
        "magic": b"PTE1",
        "expect_valid": True,
    },
    {
        "segments": [(2, 1024, 512, 16), (1, 2048, 4096, 128)],
        "version": 1,
        "magic": b"PTE1",
        "expect_valid": True,
    },
    {
        "segments": [(0, 0, 512, 16), (2, 512, 512, 16)],
        "version": 1,
        "magic": b"PTE1",
        "expect_valid": False,
    },
    {
        "segments": [(1, 1024, 2048, 64)],
        "version": 2,
        "magic": b"PTE1",
        "expect_valid": False,
    },
    {
        "segments": [(1, 1024, 2048, 64)],
        "version": 1,
        "magic": b"BAD1",
        "expect_valid": False,
    },
]

BUFFER_TEST_CASES = [
    [
        {"id": "t1", "size": 1024, "liveness": (0, 3)},
        {"id": "t2", "size": 2048, "liveness": (1, 4)},
        {"id": "t3", "size": 512, "liveness": (3, 6)},
        {"id": "t4", "size": 4096, "liveness": (5, 8)},
    ],
    [
        {"id": "conv1", "size": 8192, "liveness": (0, 2)},
        {"id": "relu1", "size": 8192, "liveness": (2, 4)},
        {"id": "conv2", "size": 16384, "liveness": (3, 6)},
        {"id": "pool1", "size": 4096, "liveness": (5, 7)},
        {"id": "fc1", "size": 2048, "liveness": (7, 9)},
    ],
    [
        {"id": "a", "size": 500, "liveness": (0, 10), "alignment": 32},
        {"id": "b", "size": 1200, "liveness": (0, 4)},
        {"id": "c", "size": 1200, "liveness": (5, 10)},
        {"id": "d", "size": 300, "liveness": (2, 8), "alignment": 128},
    ],
    [],
]
