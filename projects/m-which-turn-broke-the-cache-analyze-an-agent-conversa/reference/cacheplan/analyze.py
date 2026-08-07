def simulate_processing(prompts: list[list[str]]) -> list[int]:
    counts = []
    for i, p in enumerate(prompts):
        if i == 0:
            counts.append(len(p))
        else:
            prev = prompts[i-1]
            match_len = 0
            for a, b in zip(p, prev):
                if a == b:
                    match_len += 1
                else:
                    break
            counts.append(len(p) - match_len)
    return counts

def find_breaking_turn(counts: list[int]) -> int:
    if len(counts) <= 1:
        return -1
    return max(range(1, len(counts)), key=lambda i: counts[i])
