def optimize_chunk_size(min_size: int, max_size: int) -> int:
    best_size = min_size
    best_score = float('inf')
    for sz in range(min_size, max_size + 1, 64):
        score = abs(sz - 256)
        if score < best_score:
            best_score = score
            best_size = sz
    return best_size

def validate_chunk_size(size: int) -> bool:
    if size <= 0:
        return False
    if size % 64 != 0:
        return False
    return True
