def derive_gqa_and_head_dim(metadata):
    hidden_dim = metadata.get("embedding_length", metadata.get("hidden_size", 0))
    head_count = metadata.get("attention.head_count", 0)
    head_count_kv = metadata.get("attention.head_count_kv", head_count)
    head_dim = metadata.get("attention.head_dim", hidden_dim // head_count if head_count > 0 else 0)
    gqa_ratio = head_count // head_count_kv if head_count_kv > 0 else 1
    return {"gqa_ratio": int(gqa_ratio), "head_dim": int(head_dim)}
