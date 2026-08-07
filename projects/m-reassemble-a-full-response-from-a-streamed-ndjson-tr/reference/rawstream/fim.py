def make_fim_request(prefix, suffix, middle=""):
    if not isinstance(prefix, str) or not isinstance(suffix, str):
        raise ValueError("prefix and suffix must be strings")
    prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"
    return {
        "prompt": prompt,
        "prefix": prefix,
        "suffix": suffix,
        "middle": middle
    }
