def export_vocab_only(vocab_data):
    tokens = vocab_data["tokens"]
    scores = vocab_data["scores"]
    encoded = []
    for t, s in zip(tokens, scores):
        encoded.append({"token": t.encode("utf-8").hex(), "score": float(s)})
    return {"vocab_size": len(tokens), "entries": encoded}
