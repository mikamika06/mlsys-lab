from typing import List, Tuple, Set, Dict

def partition_ops(
    nodes: List[Tuple[int, str]],
    delegate_support: Set[str]
) -> Dict[int, bool]:
    """
    Return a mapping from node ID to a boolean indicating whether the node
    can be delegated to XNNPACK.

    The decision is simply membership of the op name in the support set.
    """
    return {node_id: op_name in delegate_support for node_id, op_name in nodes}
