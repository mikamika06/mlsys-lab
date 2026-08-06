def truncate_sequence(tokens, num_ctx, policy="truncate_right"):
    tokens = list(tokens)
    length = len(tokens)
    if length <= num_ctx:
        return {
            "tokens": tokens,
            "truncated": False,
            "original_length": length,
            "final_length": length,
            "policy_applied": policy,
        }

    if policy == "error":
        raise ValueError(f"Sequence length {length} exceeds max allowed context {num_ctx}")
    elif policy == "truncate_right":
        truncated_tokens = tokens[:num_ctx]
    elif policy == "truncate_left":
        truncated_tokens = tokens[-num_ctx:]
    else:
        raise ValueError(f"Unknown truncation policy: {policy}")

    return {
        "tokens": truncated_tokens,
        "truncated": True,
        "original_length": length,
        "final_length": num_ctx,
        "policy_applied": policy,
    }
