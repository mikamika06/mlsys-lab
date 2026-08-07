def classify_vocab_type(artifacts):
    if "model" in artifacts and artifacts["model"].get("type") == "BPE":
        return "BPE"
    if "vocab" in artifacts and isinstance(artifacts["vocab"], list):
        return "WordPiece"
    return "Unigram"
