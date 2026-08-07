HARDWARE_CONFIGS = [
    {
        0: {"capacity_bytes": 16 * 1024**3, "bandwidth_gbps": 900.0, "latency_us": 1.0},
        1: {"capacity_bytes": 64 * 1024**3, "bandwidth_gbps": 50.0, "latency_us": 10.0},
        2: {"capacity_bytes": 512 * 1024**3, "bandwidth_gbps": 3.2, "latency_us": 100.0},
    },
    {
        0: {"capacity_bytes": 8 * 1024**3, "bandwidth_gbps": 600.0, "latency_us": 2.0},
        1: {"capacity_bytes": 32 * 1024**3, "bandwidth_gbps": 40.0, "latency_us": 15.0},
        2: {"capacity_bytes": 256 * 1024**3, "bandwidth_gbps": 2.5, "latency_us": 150.0},
    },
    {
        0: {"capacity_bytes": 4 * 1024**3, "bandwidth_gbps": 800.0, "latency_us": 1.5},
        1: {"capacity_bytes": 128 * 1024**3, "bandwidth_gbps": 60.0, "latency_us": 8.0},
        2: {"capacity_bytes": 1024 * 1024**3, "bandwidth_gbps": 6.4, "latency_us": 80.0},
    },
    {
        0: {"capacity_bytes": 24 * 1024**3, "bandwidth_gbps": 1000.0, "latency_us": 1.0},
        1: {"capacity_bytes": 96 * 1024**3, "bandwidth_gbps": 80.0, "latency_us": 5.0},
        2: {"capacity_bytes": 2048 * 1024**3, "bandwidth_gbps": 7.0, "latency_us": 50.0},
    },
    {
        0: {"capacity_bytes": 2 * 1024**3, "bandwidth_gbps": 500.0, "latency_us": 3.0},
        1: {"capacity_bytes": 16 * 1024**3, "bandwidth_gbps": 30.0, "latency_us": 20.0},
        2: {"capacity_bytes": 128 * 1024**3, "bandwidth_gbps": 1.5, "latency_us": 200.0},
    },
]

TENSOR_SETS = [
    [
        {"id": f"t_{i}", "size_bytes": (i % 4 + 1) * 2 * 1024**3, "access_frequency": 10 * (10 - i)}
        for i in range(8)
    ],
    [
        {"id": f"t_{i}", "size_bytes": (i + 1) * 1024**3, "access_frequency": (i + 1) * 5}
        for i in range(6)
    ],
    [
        {"id": f"t_{i}", "size_bytes": 1024**3, "access_frequency": 20 if i < 3 else 2}
        for i in range(5)
    ],
    [
        {"id": f"t_{i}", "size_bytes": 3 * 1024**3, "access_frequency": 50 - i * 5}
        for i in range(7)
    ],
    [
        {"id": f"t_{i}", "size_bytes": (i % 2 + 1) * 512 * 1024**2, "access_frequency": 15}
        for i in range(10)
    ],
]

LOG_SAMPLES = [
    "2026-08-01 10:00:00 [NVME] READ tensor_0 size_bytes=1048576 duration_us=250 status=OK",
    "2026-08-01 10:00:01 [NVME] WRITE tensor_1 size_bytes=2097152 duration_us=600 status=OK",
    "2026-08-01 10:00:02 [NVME] PREFETCH tensor_2 size_bytes=4194304 status=HIT",
    "2026-08-01 10:00:03 [NVME] PREFETCH tensor_3 size_bytes=4194304 status=MISS",
    "2026-08-01 10:00:04 [NVME] STALL reason=nvme_queue_full duration_us=1200",
    "2026-08-01 10:00:05 [NVME] READ tensor_4 size_bytes=8388608 duration_us=1800 status=OK",
    "2026-08-01 10:00:06 [NVME] WRITE tensor_5 size_bytes=4194304 duration_us=1000 status=OK",
    "2026-08-01 10:00:07 [NVME] PREFETCH tensor_6 size_bytes=2097152 status=HIT",
    "2026-08-01 10:00:08 [NVME] STALL reason=host_buffer_busy duration_us=800",
]


def tensor_cost(tensor, target_spec):
    bw_bytes_per_sec = target_spec["bandwidth_gbps"] * 1e9
    latency_sec = target_spec["latency_us"] * 1e-6
    transfer_sec = tensor["size_bytes"] / bw_bytes_per_sec
    return (transfer_sec + latency_sec) * tensor["access_frequency"]


def select_offload_targets(tensors, hardware):
    sorted_tensors = sorted(
        tensors,
        key=lambda t: (t["access_frequency"] / t["size_bytes"], t["id"]),
        reverse=True,
    )
    used_bytes = {0: 0, 1: 0, 2: 0}
    placements = {}

    for t in sorted_tensors:
        best_target = None
        best_cost = float("inf")
        for dev in (0, 1, 2):
            if used_bytes[dev] + t["size_bytes"] <= hardware[dev]["capacity_bytes"]:
                c = tensor_cost(t, hardware[dev])
                if c < best_cost:
                    best_cost = c
                    best_target = dev
        if best_target is None:
            best_target = 2
        used_bytes[best_target] += t["size_bytes"]
        placements[t["id"]] = {
            "target": best_target,
            "cost": best_cost,
        }

    return {
        "assignments": {tid: data["target"] for tid, data in placements.items()},
        "device_usage": used_bytes,
    }


def compare_adam_performance(tensor_sizes, thread_counts):
    base_throughput = 2.5e8
    results = []
    for numel in tensor_sizes:
        for threads in thread_counts:
            torch_time = (numel / base_throughput) / (threads**0.35)
            ds_time = (numel / base_throughput) / (4.0 * (threads**0.85))
            results.append(
                {
                    "numel": numel,
                    "threads": threads,
                    "torch_time_ms": round(torch_time * 1000.0, 4),
                    "deepspeed_time_ms": round(ds_time * 1000.0, 4),
                    "speedup": round(torch_time / ds_time, 4),
                }
            )
    return results


def analyze_nvme_logs(log_lines):
    read_bytes = 0
    write_bytes = 0
    read_bw_list = []
    write_bw_list = []
    hits = 0
    misses = 0
    stall_us = 0

    for line in log_lines:
        tokens = line.strip().split()
        if "[NVME]" not in tokens:
            continue
        idx = tokens.index("[NVME]")
        if idx + 1 >= len(tokens):
            continue
        op = tokens[idx + 1]

        kv = {}
        for item in tokens[idx + 2 :]:
            if "=" in item:
                k, v = item.split("=", 1)
                kv[k] = v

        if op == "READ":
            sz = int(kv.get("size_bytes", 0))
            dur = int(kv.get("duration_us", 1))
            read_bytes += sz
            read_bw_list.append((sz / 1e9) / (dur / 1e6))
        elif op == "WRITE":
            sz = int(kv.get("size_bytes", 0))
            dur = int(kv.get("duration_us", 1))
            write_bytes += sz
            write_bw_list.append((sz / 1e9) / (dur / 1e6))
        elif op == "PREFETCH":
            st = kv.get("status", "")
            if st == "HIT":
                hits += 1
            elif st == "MISS":
                misses += 1
        elif op == "STALL":
            dur = int(kv.get("duration_us", 0))
            stall_us += dur

    avg_read_bw = sum(read_bw_list) / len(read_bw_list) if read_bw_list else 0.0
    avg_write_bw = sum(write_bw_list) / len(write_bw_list) if write_bw_list else 0.0
    total_prefetch = hits + misses
    hit_rate = (hits / total_prefetch) if total_prefetch > 0 else 0.0

    return {
        "total_read_bytes": read_bytes,
        "total_write_bytes": write_bytes,
        "avg_read_bandwidth_gbps": round(avg_read_bw, 4),
        "avg_write_bandwidth_gbps": round(avg_write_bw, 4),
        "prefetch_hit_rate": round(hit_rate, 4),
        "total_stall_time_ms": round(stall_us / 1000.0, 4),
    }
