TABLES = [
    [
        {"name": "aten::add", "self_cpu_time_total": 1200, "cpu_time_total": 1500, "input_shape": [32, 128]},
        {"name": "aten::matmul", "self_cpu_time_total": 8500, "cpu_time_total": 9000, "input_shape": [32, 128, 128]},
        {"name": "aten::gelu", "self_cpu_time_total": 500, "cpu_time_total": 600, "input_shape": [32, 128]}
    ],
    [
        {"name": "aten::mm", "self_cpu_time_total": 14000, "cpu_time_total": 15000, "input_shape": [64, 256]},
        {"name": "aten::copy_", "self_cpu_time_total": 2000, "cpu_time_total": 2000, "input_shape": [64, 256]},
        {"name": "aten::relu", "self_cpu_time_total": 1000, "cpu_time_total": 1100, "input_shape": [64, 256]}
    ],
    [
        {"name": "aten::bmm", "self_cpu_time_total": 22000, "cpu_time_total": 23000, "input_shape": [128, 64, 64]},
        {"name": "aten::addmm", "self_cpu_time_total": 5000, "cpu_time_total": 5500, "input_shape": [128, 64]},
        {"name": "aten::layer_norm", "self_cpu_time_total": 3000, "cpu_time_total": 4000, "input_shape": [128, 64]}
    ]
]

BATCH_TABLES = {
    8: [
        {"name": "aten::matmul", "self_cpu_time_total": 1000, "input_shape": [8, 64]},
        {"name": "aten::add", "self_cpu_time_total": 200, "input_shape": [8, 64]}
    ],
    16: [
        {"name": "aten::matmul", "self_cpu_time_total": 2000, "input_shape": [16, 64]},
        {"name": "aten::matmul", "self_cpu_time_total": 500, "input_shape": [16, 32]},
        {"name": "aten::add", "self_cpu_time_total": 400, "input_shape": [16, 64]}
    ],
    32: [
        {"name": "aten::matmul", "self_cpu_time_total": 4000, "input_shape": [32, 64]},
        {"name": "aten::matmul", "self_cpu_time_total": 1000, "input_shape": [32, 32]},
        {"name": "aten::matmul", "self_cpu_time_total": 300, "input_shape": [32, 16]},
        {"name": "aten::add", "self_cpu_time_total": 800, "input_shape": [32, 64]}
    ]
}


def top_1_operator(records):
    best = max(records, key=lambda x: x["self_cpu_time_total"])
    return best["name"]


def matmul_share(records):
    matmul_prefixes = ("aten::matmul", "aten::mm", "aten::bmm", "aten::addmm")
    total = sum(r["self_cpu_time_total"] for r in records)
    if total == 0:
        return 0.0
    mm_total = sum(r["self_cpu_time_total"] for r in records if r["name"].startswith(matmul_prefixes))
    return mm_total / total


def row_count_delta(batch_tables):
    counts = {b: len(recs) for b, recs in batch_tables.items()}
    sorted_batches = sorted(counts.keys())
    deltas = [counts[sorted_batches[i+1]] - counts[sorted_batches[i]] for i in range(len(sorted_batches) - 1)]
    return sum(deltas)
