def optimize_prompt_layout(components, block_size):
    stable = [c for c in components if c.get("stable", False)]
    volatile = [c for c in components if not c.get("stable", False)]

    reordered = stable + volatile
    optimized_tokens = []
    for c in reordered:
        optimized_tokens.extend(c["tokens"])

    full_block_tokens = (len(optimized_tokens) // block_size) * block_size
    prefix_tokens = optimized_tokens[:full_block_tokens]

    return {
        "optimized_components": reordered,
        "prompt_tokens": optimized_tokens,
        "prefix_block_tokens": prefix_tokens,
    }
