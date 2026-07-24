from mlsys.sim import GPU


def _make_cases(num_cases, max_edges):
    data = []
    for c in range(num_cases):
        for e in range(max_edges):
            size = 8 + ((17 * c + 11 * e + 5) % 97)
            reuse = 1 + ((5 * c + 7 * e + 3) % 5)
            recompute = 1 + ((3 * c * c + 13 * e + c * e + 19) % 80)

            if (c + 2 * e) % 11 == 0:
                reuse = 2 + ((c + e) % 4)
                recompute = (2 * size) // (reuse - 1)

            data.extend([size, reuse, recompute])
    return data


def _reference(flat_graph, num_cases, max_edges):
    out = []
    for c in range(num_cases):
        for e in range(max_edges):
            off = (c * max_edges + e) * 3
            size = int(flat_graph[off])
            reuse = int(flat_graph[off + 1])
            recompute = int(flat_graph[off + 2])
            cut_cost = 2 * size
            fuse_cost = recompute * (reuse - 1)
            out.append(1 if cut_cost <= fuse_cost else 0)
    return out


def grade(sol, fx) -> dict:
    num_cases = 24
    max_edges = 6
    flat_graph = _make_cases(num_cases, max_edges)
    expected = _reference(flat_graph, num_cases, max_edges)

    graph_base = 0
    out_base = len(flat_graph) + 16
    total_edges = num_cases * max_edges
    gmem_size = out_base + total_edges + 16

    try:
        g = GPU(gmem_size)
        for i, value in enumerate(flat_graph):
            g.gmem[graph_base + i] = int(value)

        block = 32
        grid = (total_edges + block - 1) // block
        g.launch(
            sol.fusion_boundary_kernel,
            grid,
            block,
            graph_base,
            out_base,
            num_cases,
            max_edges,
        )

        got = [int(g.gmem[out_base + i]) for i in range(total_edges)]
        ok = 1.0 if got == expected else 0.0
    except Exception:
        ok = 0.0

    return {"exact_match": ok}
