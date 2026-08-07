def warp_divergence_branch_count(preds: list[int], warp_size: int = 32) -> list[int]:
    if not isinstance(preds, list) or any(isinstance(item, list) for item in preds):
        raise ValueError("preds must be a one-dimensional list")
    n = len(preds)
    if n % warp_size != 0:
        raise ValueError(f"Length {n} is not a multiple of warp_size {warp_size}")
    num_blocks = n // warp_size
    out = [0] * num_blocks
    for i in range(num_blocks):
        block = preds[i * warp_size : (i + 1) * warp_size]
        unique_count = 0
        seen = []
        for item in block:
            is_new = True
            for s in seen:
                if s == item:
                    is_new = False
                    break
            if is_new:
                seen.append(item)
                unique_count += 1
        out[i] = unique_count
    return out
