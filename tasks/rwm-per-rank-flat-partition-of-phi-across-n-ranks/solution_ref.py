def partition_phi(phi, n_ranks, param_bytes, grad_bytes, opt_bytes):
    base = phi // n_ranks
    remainder = phi % n_ranks

    result = []
    for rank in range(n_ranks):
        params = base + (1 if rank < remainder else 0)
        result.append((
            params * param_bytes,
            params * grad_bytes,
            params * opt_bytes,
        ))
    return result
