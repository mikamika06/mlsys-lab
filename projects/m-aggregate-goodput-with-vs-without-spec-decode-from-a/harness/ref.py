import random

OUTCOMES = [
    "TP_RANK_MISMATCH",
    "DRAFT_HEAD_SHAPE_MISMATCH",
    "IPC_MEM_HANDLE_LEAK",
    "TREE_MASK_BUFFER_OVERFLOW",
    "VOCAB_SIZE_OUT_OF_BOUNDS",
    "NCCL_TIMEOUT_DEADLOCK",
]


def make_diagnostic_payloads(seed=42):
    rng = random.Random(seed)
    payloads = []

    # 0: TP_RANK_MISMATCH
    payloads.append({
        "tp_world_size": 4,
        "ranks": [0, 1, 2, 2],
        "base_hidden_dim": 4096,
        "draft_hidden_dim": 4096,
        "open_ipc_handles": 5,
        "max_ipc_handles": 64,
        "tree_mask_size": 1024,
        "tree_width": 16,
        "max_tree_depth": 64,
        "base_vocab_size": 32000,
        "draft_vocab_size": 32000,
        "barrier_timeout": False,
    })

    # 1: DRAFT_HEAD_SHAPE_MISMATCH
    payloads.append({
        "tp_world_size": 2,
        "ranks": [0, 1],
        "base_hidden_dim": 4096,
        "draft_hidden_dim": 2048,
        "open_ipc_handles": 10,
        "max_ipc_handles": 64,
        "tree_mask_size": 1024,
        "tree_width": 16,
        "max_tree_depth": 64,
        "base_vocab_size": 32000,
        "draft_vocab_size": 32000,
        "barrier_timeout": False,
    })

    # 2: IPC_MEM_HANDLE_LEAK
    payloads.append({
        "tp_world_size": 8,
        "ranks": list(range(8)),
        "base_hidden_dim": 8192,
        "draft_hidden_dim": 8192,
        "open_ipc_handles": 128,
        "max_ipc_handles": 64,
        "tree_mask_size": 2048,
        "tree_width": 16,
        "max_tree_depth": 64,
        "base_vocab_size": 128000,
        "draft_vocab_size": 128000,
        "barrier_timeout": False,
    })

    # 3: TREE_MASK_BUFFER_OVERFLOW
    payloads.append({
        "tp_world_size": 4,
        "ranks": list(range(4)),
        "base_hidden_dim": 4096,
        "draft_hidden_dim": 4096,
        "open_ipc_handles": 16,
        "max_ipc_handles": 64,
        "tree_mask_size": 256,
        "tree_width": 16,
        "max_tree_depth": 64,
        "base_vocab_size": 32000,
        "draft_vocab_size": 32000,
        "barrier_timeout": False,
    })

    # 4: VOCAB_SIZE_OUT_OF_BOUNDS
    payloads.append({
        "tp_world_size": 2,
        "ranks": [0, 1],
        "base_hidden_dim": 4096,
        "draft_hidden_dim": 4096,
        "open_ipc_handles": 8,
        "max_ipc_handles": 64,
        "tree_mask_size": 1024,
        "tree_width": 16,
        "max_tree_depth": 64,
        "base_vocab_size": 32000,
        "draft_vocab_size": 32003,
        "barrier_timeout": False,
    })

    # 5: NCCL_TIMEOUT_DEADLOCK
    payloads.append({
        "tp_world_size": 4,
        "ranks": list(range(4)),
        "base_hidden_dim": 4096,
        "draft_hidden_dim": 4096,
        "open_ipc_handles": 4,
        "max_ipc_handles": 64,
        "tree_mask_size": 1024,
        "tree_width": 16,
        "max_tree_depth": 64,
        "base_vocab_size": 32000,
        "draft_vocab_size": 32000,
        "barrier_timeout": True,
    })

    return payloads


def make_scheduler_logs(num_entries=50, seed=123):
    rng = random.Random(seed)
    log = []

    for i in range(num_entries):
        is_spec = i % 2 == 0
        sla_lat = rng.randint(100, 300)

        if is_spec:
            lat = rng.randint(80, 250)
            dur = rng.randint(500, 2000)
            accepted = rng.randint(50, 200)
            rejected = rng.randint(5, 40)
            entry = {
                "request_id": f"req_{i}",
                "is_speculative": True,
                "latency_ms": lat,
                "sla_latency_ms": sla_lat,
                "duration_ms": dur,
                "accepted_tokens": accepted,
                "rejected_tokens": rejected,
            }
        else:
            lat = rng.randint(100, 280)
            dur = rng.randint(600, 2200)
            generated = rng.randint(40, 150)
            entry = {
                "request_id": f"req_{i}",
                "is_speculative": False,
                "latency_ms": lat,
                "sla_latency_ms": sla_lat,
                "duration_ms": dur,
                "generated_tokens": generated,
            }
        log.append(entry)

    return log
