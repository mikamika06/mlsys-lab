import numpy as np

PAGE_SIZE = 4096

FIXTURES_M1 = [
    {
        "file_size": 1024 * 1024 * 100,
        "accesses": [(0, 100), (4000, 200), (8192, 4096)],
    },
    {
        "file_size": 1024 * 1024 * 500,
        "accesses": [(500, 1000), (12000, 5000), (100000, 50)],
    },
    {
        "file_size": 1024 * 1024 * 1024,
        "accesses": [(i * 3000, 1500) for i in range(100)],
    },
]

FIXTURES_M2 = [
    {
        "file_size": 1024 * 1024 * 64,
        "read_events": [
            [(0, 2000), (4000, 1000)],
            [(8000, 100)],
            [(0, 1000)],
        ],
        "evict_pages": [1],
    },
    {
        "file_size": 1024 * 1024 * 128,
        "read_events": [
            [(i * 8192, 1000) for i in range(10)],
            [(i * 4096 + 2000, 500) for i in range(5)],
        ],
        "evict_pages": [0, 2, 4],
    },
]


def estimate_resident_bytes(file_size, accesses, page_size=PAGE_SIZE):
    resident_pages = set()
    for offset, length in accesses:
        if length <= 0 or offset >= file_size:
            continue
        end_offset = min(offset + length, file_size)
        start_page = offset // page_size
        end_page = (end_offset - 1) // page_size
        for page_idx in range(start_page, end_page + 1):
            resident_pages.add(page_idx)
    return len(resident_pages) * page_size


def simulate_cache_tracker(file_size, read_events, evict_pages, page_size=PAGE_SIZE):
    resident = set()
    history = []
    max_pages = (file_size + page_size - 1) // page_size

    for accesses in read_events:
        for offset, length in accesses:
            if length <= 0 or offset >= file_size:
                continue
            end_offset = min(offset + length, file_size)
            start_p = offset // page_size
            end_p = (end_offset - 1) // page_size
            for p in range(start_p, end_p + 1):
                if p < max_pages:
                    resident.add(p)
        history.append(len(resident) * page_size)

    for p in evict_pages:
        resident.discard(p)
    history.append(len(resident) * page_size)

    return history
