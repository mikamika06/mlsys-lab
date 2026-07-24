def verify_medusa_candidates(candidates, target_probs, threshold):
    accepted_indices = []
    best_index = 0
    best_len = -1
    best_path = []

    for idx, path in enumerate(candidates):
        accepted_path = []
        for pos, token in enumerate(path):
            if pos >= len(target_probs):
                break
            if target_probs[pos].get(token, 0.0) < threshold:
                break
            accepted_path.append(token)

        if accepted_path:
            accepted_indices.append(idx)

        if len(accepted_path) > best_len:
            best_len = len(accepted_path)
            best_index = idx
            best_path = accepted_path

    return best_index, best_path, accepted_indices
