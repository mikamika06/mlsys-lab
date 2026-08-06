import ref


def check(workdir):
    from inductorsched.fusion import greedy_pointwise_fuse
    from inductorsched.memory import compute_memory_usage
    import inductorsched.fusion as ref_fusion
    import inductorsched.memory as ref_memory

    samples = ref.get_graph_samples()
    schedules_ok = True
    memory_ok = True

    for graph in samples:
        ref_fused = ref_fusion.greedy_pointwise_fuse(graph)
        got_fused = greedy_pointwise_fuse(graph)
        if got_fused != ref_fused:
            schedules_ok = False

        ref_mem_off = ref_memory.compute_memory_usage(graph, ref_fused, inplace_buffers=False)
        ref_mem_on = ref_memory.compute_memory_usage(graph, ref_fused, inplace_buffers=True)

        got_mem_off = compute_memory_usage(graph, got_fused, inplace_buffers=False)
        got_mem_on = compute_memory_usage(graph, got_fused, inplace_buffers=True)

        if got_mem_off != ref_mem_off or got_mem_on != ref_mem_on:
            memory_ok = False

    return {
        "schedules_matched": 1.0 if schedules_ok else 0.0,
        "memory_savings_matched": 1.0 if memory_ok else 0.0,
    }
