def overlappable_ops(types: list[str], edges: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """
    Return every (comm_id, compute_id) pair such that neither op is a
    transitive prerequisite of the other in the dependency DAG described
    by `edges` (u, v) = "u must finish before v starts".
    """
    raise NotImplementedError('your code here')
