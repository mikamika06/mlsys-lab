def detect_leakage(mask: list[list[float]], cu_seqlens: list[int]) -> bool:
    N = len(mask)
    seg_ids = [0] * N
    for i in range(len(cu_seqlens)-1):
        for idx in range(cu_seqlens[i], cu_seqlens[i+1]):
            seg_ids[idx] = i

    for i in range(N):
        for j in range(N):
            if (seg_ids[i] != seg_ids[j]) and bool(mask[i][j]):
                return True
    return False
