def compute_memory_usage(nodes, fused_kernels, inplace_buffers=False):
    import math
    node_map = {n['id']: n for n in nodes}

    def size_of(n):
        return math.prod(n['shape']) * 4

    if not inplace_buffers:
        total = sum(size_of(n) for n in nodes)
        return total

    kernel_map = {}
    for k_idx, k_nodes in enumerate(fused_kernels):
        for nid in k_nodes:
            kernel_map[nid] = k_idx

    last_user = {}
    for k_idx, k_nodes in enumerate(fused_kernels):
        for nid in k_nodes:
            n = node_map[nid]
            for inp in n['inputs']:
                last_user[inp] = max(last_user.get(inp, -1), k_idx)

    free_buffers = {}
    allocated_buffers = {}
    peak_memory = 0
    current_memory = 0

    for k_idx, k_nodes in enumerate(fused_kernels):
        for nid in k_nodes:
            n = node_map[nid]
            sz = size_of(n)

            reused = False
            for inp_id in n['inputs']:
                if inp_id in node_map and last_user.get(inp_id) == k_idx:
                    inp_sz = size_of(node_map[inp_id])
                    if inp_sz >= sz and free_buffers.get(inp_id, False):
                        allocated_buffers[nid] = inp_sz
                        free_buffers[inp_id] = False
                        reused = True
                        break

            if not reused:
                allocated_buffers[nid] = sz
                current_memory += sz
                if current_memory > peak_memory:
                    peak_memory = current_memory

            for inp_id in n['inputs']:
                if inp_id in node_map and last_user.get(inp_id) == k_idx:
                    free_buffers[inp_id] = True

    return peak_memory
