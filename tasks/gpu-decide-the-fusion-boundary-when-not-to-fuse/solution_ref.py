def fusion_boundary_kernel(t, graph_base, out_base, num_cases, max_edges):
    gid = t.gid
    total = num_cases * max_edges
    if gid >= total:
        return

    off = graph_base + gid * 3
    size = t.gload(off)
    reuse = t.gload(off + 1)
    recompute = t.gload(off + 2)

    cut_cost = 2 * size
    fuse_cost = recompute * (reuse - 1)
    t.alu(4)

    decision = 1 if cut_cost <= fuse_cost else 0
    t.gstore(out_base + gid, decision)
