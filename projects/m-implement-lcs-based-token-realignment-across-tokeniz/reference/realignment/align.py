def align_tokens(draft_tokens, target_tokens):
    n = len(draft_tokens)
    m = len(target_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if draft_tokens[i - 1] == target_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    i, j = n, m
    mapping = []
    while i > 0 and j > 0:
        if draft_tokens[i - 1] == target_tokens[j - 1]:
            mapping.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    mapping.reverse()
    return mapping
