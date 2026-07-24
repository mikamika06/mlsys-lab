def _lcp(q, c):
    n = min(len(q), len(c))
    i = 0
    while i < n and q[i] == c[i]:
        i += 1
    return i


def compute_reuse_savings(seqs, block_size):
    radix_total = 0
    block_total = 0
    for i in range(len(seqs)):
        best = 0
        for j in range(i):
            lcp = _lcp(seqs[i], seqs[j])
            if lcp > best:
                best = lcp
        radix_total += best
        block_total += (best // block_size) * block_size
    return radix_total, block_total
