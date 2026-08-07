def classify_vocab_type(artifacts):
    has_merges = artifacts.get("has_merges", False)
    has_scores = artifacts.get("has_scores", False)
    if has_merges and not has_scores:
        return "BPE"
    if has_scores and not has_merges:
        return "SentencePiece"
    if has_merges and has_scores:
        return "WPM"
    return "Unigram"

def find_wrong_token_type(tokens):
    counts = {}
    for t in tokens:
        tt = t.get("token_type")
        counts[tt] = counts.get(tt, 0) + 1
    if not counts:
        return None
    expected_majority = max(counts, key=counts.get)
    for t in tokens:
        if t.get("token_type") != expected_majority:
            return t.get("id")
    return None
