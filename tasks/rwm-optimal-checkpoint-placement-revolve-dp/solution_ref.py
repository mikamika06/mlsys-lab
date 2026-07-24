def optimal_recompute(L: int, C: int) -> int:
    if L <= 1:
        return 0

    if C >= L:
        return 0

    dp = [[0] * (C + 1) for _ in range(L + 1)]

    for layers in range(2, L + 1):
        dp[layers][0] = layers * (layers - 1) // 2

    for checkpoints in range(1, C + 1):
        for layers in range(2, L + 1):
            if checkpoints >= layers:
                dp[layers][checkpoints] = 0
                continue
            best = None
            for split in range(1, layers):
                value = (
                    split
                    + dp[split][checkpoints]
                    + dp[layers - split][checkpoints - 1]
                )
                if best is None or value < best:
                    best = value
            dp[layers][checkpoints] = best

    return int(dp[L][C])
