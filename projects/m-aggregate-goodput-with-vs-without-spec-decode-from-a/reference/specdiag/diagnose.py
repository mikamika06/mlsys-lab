def diagnose_startup(payload: dict) -> str:
    tp_world = payload.get("tp_world_size", 1)
    ranks = payload.get("ranks", [])
    if len(ranks) != tp_world or set(ranks) != set(range(tp_world)):
        return "TP_RANK_MISMATCH"

    base_dim = payload.get("base_hidden_dim")
    draft_dim = payload.get("draft_hidden_dim")
    if base_dim is not None and draft_dim is not None and base_dim != draft_dim:
        return "DRAFT_HEAD_SHAPE_MISMATCH"

    open_ipc = payload.get("open_ipc_handles", 0)
    max_ipc = payload.get("max_ipc_handles", 0)
    if open_ipc > max_ipc:
        return "IPC_MEM_HANDLE_LEAK"

    tree_size = payload.get("tree_mask_size", 0)
    max_tree_nodes = payload.get("tree_width", 0) * payload.get("max_tree_depth", 0)
    if tree_size < max_tree_nodes:
        return "TREE_MASK_BUFFER_OVERFLOW"

    base_vocab = payload.get("base_vocab_size")
    draft_vocab = payload.get("draft_vocab_size")
    if base_vocab is not None and draft_vocab is not None and base_vocab != draft_vocab:
        return "VOCAB_SIZE_OUT_OF_BOUNDS"

    if payload.get("barrier_timeout", False):
        return "NCCL_TIMEOUT_DEADLOCK"

    return "HEALTHY"


def diagnose_all_outcomes(payloads: list[dict]) -> list[str]:
    return [diagnose_startup(p) for p in payloads]
