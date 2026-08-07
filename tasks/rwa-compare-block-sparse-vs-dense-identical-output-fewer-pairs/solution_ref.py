from __future__ import annotations

import math


def block_sparse_attention(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
    mask: list[list[bool]],
    block_size: int,
) -> tuple[list[list[float]], int]:
    n = len(Q)
    d = len(Q[0])
    m = len(V[0])

    scores = [[0.0 for _ in range(n)] for _ in range(n)]
    scale = math.sqrt(d)
    for i in range(n):
        for j in range(n):
            dot = 0.0
            for k_idx in range(d):
                dot += Q[i][k_idx] * K[j][k_idx]
            scores[i][j] = dot / scale

    row_max = [-float("inf") for _ in range(n)]
    attended_pairs = 0

    for bi in range(0, n, block_size):
        for bj in range(0, n, block_size):
            b_h = min(block_size, n - bi)
            b_w = min(block_size, n - bj)

            has_any = False
            for r in range(b_h):
                for c in range(b_w):
                    if mask[bi + r][bj + c]:
                        has_any = True
                        break
                if has_any:
                    break

            if has_any:
                attended_pairs += b_h * b_w
                for li in range(b_h):
                    i_idx = bi + li
                    for lj in range(b_w):
                        j_idx = bj + lj
                        if mask[i_idx][j_idx]:
                            val = scores[i_idx][j_idx]
                            if val > row_max[i_idx]:
                                row_max[i_idx] = val

    denom = [0.0 for _ in range(n)]
    numer = [[0.0 for _ in range(m)] for _ in range(n)]

    for bi in range(0, n, block_size):
        for bj in range(0, n, block_size):
            b_h = min(block_size, n - bi)
            b_w = min(block_size, n - bj)

            has_any = False
            for r in range(b_h):
                for c in range(b_w):
                    if mask[bi + r][bj + c]:
                        has_any = True
                        break
                if has_any:
                    break

            if not has_any:
                continue

            for li in range(b_h):
                i_idx = bi + li

                row_allowed = False
                for lj in range(b_w):
                    if mask[i_idx][bj + lj]:
                        row_allowed = True
                        break

                if row_allowed:
                    for lj in range(b_w):
                        j_idx = bj + lj
                        if mask[i_idx][j_idx]:
                            w = math.exp(scores[i_idx][j_idx] - row_max[i_idx])
                            denom[i_idx] += w
                            for col_v in range(m):
                                numer[i_idx][col_v] += w * V[j_idx][col_v]

    out = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        if denom[i] != 0:
            for col_v in range(m):
                out[i][col_v] = numer[i][col_v] / denom[i]

    return out, attended_pairs
