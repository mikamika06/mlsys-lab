def preempt_recompute(requests, preempt_ids):
    p_ids = set(preempt_ids)
    updated = []
    reexecuted_tokens = 0
    for req in requests:
        r = dict(req)
        if r["req_id"] in p_ids and r.get("status") == "RUNNING":
            r["status"] = "PREEMPTED"
            r["num_blocks"] = 0
            reexecuted_tokens += r.get("prompt_len", 0) + r.get("generated_len", 0)
        updated.append(r)
    return updated, reexecuted_tokens


def compute_swap_cost(num_blocks, block_bytes, pcie_bandwidth_gbps, roundtrip=True):
    direction_factor = 2 if roundtrip else 1
    bytes_moved = int(num_blocks * block_bytes * direction_factor)
    bandwidth_bytes_per_sec = pcie_bandwidth_gbps * 1e9
    time_seconds = bytes_moved / bandwidth_bytes_per_sec
    return {"bytes_moved": bytes_moved, "time_seconds": time_seconds}


def choose_preemption_mode(workload_profile):
    recompute_tokens = workload_profile["recompute_tokens"]
    tps = workload_profile["token_processing_rate_tps"]
    recompute_time = recompute_tokens / tps

    num_blocks = workload_profile["num_blocks"]
    block_bytes = workload_profile["block_bytes"]
    pcie_bw = workload_profile["pcie_bandwidth_gbps"]
    roundtrip = workload_profile.get("roundtrip", True)

    swap_res = compute_swap_cost(num_blocks, block_bytes, pcie_bw, roundtrip=roundtrip)
    swap_time = swap_res["time_seconds"]

    if swap_time < recompute_time:
        return "swap"
    return "recompute"


REQUEST_WORKLOADS = [
    {
        "requests": [
            {"req_id": "req_0", "prompt_len": 512, "generated_len": 64, "num_blocks": 36, "status": "RUNNING"},
            {"req_id": "req_1", "prompt_len": 1024, "generated_len": 128, "num_blocks": 72, "status": "RUNNING"},
            {"req_id": "req_2", "prompt_len": 256, "generated_len": 16, "num_blocks": 18, "status": "WAITING"},
        ],
        "preempt_ids": ["req_0", "req_2"],
    },
    {
        "requests": [
            {"req_id": "a1", "prompt_len": 2048, "generated_len": 512, "num_blocks": 160, "status": "RUNNING"},
            {"req_id": "a2", "prompt_len": 4096, "generated_len": 1024, "num_blocks": 320, "status": "RUNNING"},
        ],
        "preempt_ids": ["a1", "a2"],
    },
    {
        "requests": [
            {"req_id": "b1", "prompt_len": 128, "generated_len": 0, "num_blocks": 8, "status": "RUNNING"},
        ],
        "preempt_ids": ["b1"],
    },
]

WORKLOAD_PROFILES = [
    {
        "num_blocks": 500,
        "block_bytes": 2 * 1024 * 1024,
        "recompute_tokens": 200,
        "token_processing_rate_tps": 20000.0,
        "pcie_bandwidth_gbps": 8.0,
        "roundtrip": True,
    },
    {
        "num_blocks": 50,
        "block_bytes": 1024 * 1024,
        "recompute_tokens": 16384,
        "token_processing_rate_tps": 5000.0,
        "pcie_bandwidth_gbps": 64.0,
        "roundtrip": True,
    },
    {
        "num_blocks": 100,
        "block_bytes": 512 * 1024,
        "recompute_tokens": 4096,
        "token_processing_rate_tps": 1000.0,
        "pcie_bandwidth_gbps": 32.0,
        "roundtrip": True,
    },
    {
        "num_blocks": 200,
        "block_bytes": 4 * 1024 * 1024,
        "recompute_tokens": 64,
        "token_processing_rate_tps": 10000.0,
        "pcie_bandwidth_gbps": 16.0,
        "roundtrip": True,
    },
    {
        "num_blocks": 128,
        "block_bytes": 1024 * 1024,
        "recompute_tokens": 8192,
        "token_processing_rate_tps": 2000.0,
        "pcie_bandwidth_gbps": 32.0,
        "roundtrip": True,
    },
    {
        "num_blocks": 1000,
        "block_bytes": 2 * 1024 * 1024,
        "recompute_tokens": 128,
        "token_processing_rate_tps": 15000.0,
        "pcie_bandwidth_gbps": 16.0,
        "roundtrip": True,
    },
]
