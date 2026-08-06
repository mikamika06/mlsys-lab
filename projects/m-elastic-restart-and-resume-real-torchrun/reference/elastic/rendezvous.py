def compute_membership(old_world_size: int, failed_ranks: list, new_world_size: int) -> dict:
    active_ranks = [i for i in range(old_world_size) if i not in failed_ranks]
    new_ranks = active_ranks[:new_world_size]
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(new_ranks)}
    return {"active_ranks": new_ranks, "mapping": mapping, "world_size": len(new_ranks)}
