def classify_tokenizer_compatibility(draft_vocab, target_vocab):
    if draft_vocab == target_vocab:
        return "identical"
    draft_set = set(draft_vocab)
    target_set = set(target_vocab)
    if draft_set.issubset(target_set):
        return "compatible_subset"
    return "cross_tokenizer"
