def chunk_prompt(tokens, chunk_size):
    if not tokens:
        return []
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
