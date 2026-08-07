import ref


def check(workdir):
    from router.routing import select_best_node

    out = {"selection_matched": 0.0}
    want = ref.select_best_node(ref.NODES_STATE, ref.REQUEST, ref.COST)
    got = select_best_node(ref.NODES_STATE, ref.REQUEST, ref.COST)
    if got == want:
        out["selection_matched"] = 1.0
    else:
        out["_note"] = f"expected best node {want}, got {got}"
    return out
