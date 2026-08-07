def check_cache_salt_isolation(salt_a, salt_b, prompt_tokens):
    hash_a = hash((salt_a, tuple(prompt_tokens)))
    hash_b = hash((salt_b, tuple(prompt_tokens)))
    return hash_a != hash_b
