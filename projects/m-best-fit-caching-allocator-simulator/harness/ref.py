from allocator.metrics import run_trace
from allocator.simulator import CachingAllocator

TRACES = [
    [
        ("alloc", "a", 512),
        ("alloc", "b", 1024),
        ("free", "a"),
        ("alloc", "c", 256),
        ("free", "b"),
        ("free", "c"),
    ],
    [
        ("alloc", "a", 3000000),
        ("alloc", "b", 1000000),
        ("free", "a"),
        ("alloc", "c", 500000),
        ("alloc", "d", 2500000),
        ("free", "b"),
        ("free", "c"),
        ("free", "d"),
    ],
    [("alloc", f"t{i}", 100000 * ((i % 5) + 1)) for i in range(20)]
    + [("free", f"t{i}") for i in range(0, 20, 2)]
    + [("alloc", f"n{i}", 150000) for i in range(10)]
    + [("free", f"t{i}") for i in range(1, 20, 2)]
    + [("free", f"n{i}") for i in range(10)],
    [
        ("alloc", "m1", 2097152),
        ("alloc", "m2", 2097152),
        ("free", "m1"),
        ("alloc", "m3", 1048576),
        ("alloc", "m4", 1048576),
        ("free", "m2"),
        ("free", "m3"),
        ("free", "m4"),
    ],
    [
        ("alloc", "x1", 5000000),
        ("free", "x1"),
        ("alloc", "x2", 4000000),
        ("free", "x2"),
    ],
]


def get_ref_metrics(trace_idx, segment_size=2097152):
    return run_trace(TRACES[trace_idx], segment_size=segment_size)
