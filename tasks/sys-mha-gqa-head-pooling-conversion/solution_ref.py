def mha_to_gqa_pool(
    K: list[list[list[list[float]]]],
    V: list[list[list[list[float]]]],
    n_kv_heads: int,
) -> tuple[list[list[list[list[float]]]], list[list[list[list[float]]]]]:
    """
    Convert MHA key/value tensors to GQA-init key/value tensors by
    mean-pooling contiguous groups of original KV heads.

    K, V: (B, H, T, D) nested lists of float MHA key/value tensors.
    n_kv_heads: target number of GQA KV heads G (H must be divisible by G).

    Heads are grouped contiguously in index order: heads
    [0, H/G) -> group 0, [H/G, 2H/G) -> group 1, etc.

    Returns (K_gqa, V_gqa), each of shape (B, G, T, D).
    """
    B = len(K)
    H = len(K[0])
    T = len(K[0][0])
    D = len(K[0][0][0])
    r = H // n_kv_heads

    K_gqa = [[[[0.0 for _ in range(D)] for _ in range(T)] for _ in range(n_kv_heads)] for _ in range(B)]
    V_gqa = [[[[0.0 for _ in range(D)] for _ in range(T)] for _ in range(n_kv_heads)] for _ in range(B)]

    for b in range(B):
        for g in range(n_kv_heads):
            for t in range(T):
                for d in range(D):
                    k_sum = 0.0
                    v_sum = 0.0
                    for i in range(r):
                        h = g * r + i
                        k_sum += K[b][h][t][d]
                        v_sum += V[b][h][t][d]
                    K_gqa[b][g][t][d] = k_sum / r
                    V_gqa[b][g][t][d] = v_sum / r

    return K_gqa, V_gqa
