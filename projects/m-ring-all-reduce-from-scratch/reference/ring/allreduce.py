import numpy as np

def ring_allreduce(arrays):
    num_ranks = len(arrays)
    shape = arrays[0].shape
    dtype = arrays[0].dtype
    flattened = [a.flatten() for a in arrays]
    total_elements = flattened[0].size
    chunk_size = (total_elements + num_ranks - 1) // num_ranks

    chunks = []
    for a in flattened:
        padded = np.pad(a, (0, num_ranks * chunk_size - total_elements), 'constant')
        chunks.append(np.split(padded, num_ranks))

    current = [c.copy() for c in chunks]

    for step in range(num_ranks - 1):
        next_current = [c.copy() for c in current]
        for r in range(num_ranks):
            recv_idx = (r - step - 1) % num_ranks
            dest = (r + 1) % num_ranks
            next_current[dest][recv_idx] = current[dest][recv_idx] + current[r][recv_idx]
        current = next_current

    reduced_chunks = [current[r][r] for r in range(num_ranks)]

    final_chunks = [rc.copy() for rc in reduced_chunks]
    for step in range(num_ranks - 1):
        next_final = [fc.copy() for fc in final_chunks]
        for r in range(num_ranks):
            send_chunk_idx = (r - step + 1) % num_ranks
            dest = (r + 1) % num_ranks
            src = (r - 1) % num_ranks
            next_final[dest][send_chunk_idx] = final_chunks[src][send_chunk_idx]
        final_chunks = next_final

    results = []
    for r in range(num_ranks):
        recon = np.concatenate(final_chunks[r])[:total_elements]
        results.append(recon.reshape(shape).astype(dtype))
    return results
