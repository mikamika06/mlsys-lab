def compare_batching_throughput(gen_lens, batch_size):
    n = len(gen_lens)

    makespan_static = 0.0
    for i in range(0, n, batch_size):
        makespan_static += max(gen_lens[i:i + batch_size])

    slot_free = [0.0] * batch_size
    for length in gen_lens:
        j = min(range(batch_size), key=lambda x: slot_free[x])
        slot_free[j] += length
    makespan_cont = max(slot_free)

    throughput_ratio = makespan_static / makespan_cont
    return makespan_static, makespan_cont, throughput_ratio
