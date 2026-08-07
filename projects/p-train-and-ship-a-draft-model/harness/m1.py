def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    m = {"has_weights": 0.0, "valid_probs": 0.0}
    try:
        from spec.draft import DraftModel
        d = DraftModel(ref.get_vocab_size(), ref.get_hidden_size())
        if hasattr(d, 'W1') and hasattr(d, 'W2'):
            m["has_weights"] = 1.0
        p = d.get_probs(0)
        if len(p) == ref.get_vocab_size() and abs(sum(p) - 1.0) < 1e-4:
            m["valid_probs"] = 1.0
    except Exception:
        pass
    return m
