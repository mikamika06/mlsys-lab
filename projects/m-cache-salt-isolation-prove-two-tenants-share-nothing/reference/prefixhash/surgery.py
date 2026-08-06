def optimize_template(prompt_tokens, timestamp_token):
    if timestamp_token in prompt_tokens:
        filtered = [t for t in prompt_tokens if t != timestamp_token]
        return filtered + [timestamp_token]
    return list(prompt_tokens)
