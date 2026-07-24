def blocked_access_trace(n: int, tile: int) -> list[int]:
    out = []
    for tr in range(0, n, tile):
        for tc in range(0, n, tile):
            for i in range(tr, min(tr + tile, n)):
                for j in range(tc, min(tc + tile, n)):
                    out.append((i * n + j) * 8)
    return out
