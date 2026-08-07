def greedy_speculative(draft_logits: list[list[float]],
                       target_logits: list[list[float]]) -> list[int]:
    """
    Return the token indices chosen by speculative decoding.
    draft_logits and target_logits are 2‑D lists of shape (T, V).
    """
    T = len(draft_logits)
    V = len(draft_logits[0])
    result = []
    for t in range(T):
        best_draft_idx = 0
        best_draft_val = draft_logits[t][0]
        for v in range(1, V):
            val = draft_logits[t][v]
            if val > best_draft_val:
                best_draft_val = val
                best_draft_idx = v

        best_target_idx = 0
        best_target_val = target_logits[t][0]
        for v in range(1, V):
            val = target_logits[t][v]
            if val > best_target_val:
                best_target_val = val
                best_target_idx = v

        if best_draft_idx == best_target_idx:
            result.append(int(best_draft_idx))
        else:
            result.append(int(best_target_idx))

    return result
