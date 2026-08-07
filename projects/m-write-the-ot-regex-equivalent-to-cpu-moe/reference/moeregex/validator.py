import re


def validate_override(regex_pattern, model_config):
    pat = re.compile(regex_pattern)
    matched = []
    for t in model_config.get("tensors", []):
        if pat.search(t):
            matched.append(t)
    return {"matched_count": len(matched), "valid": True}
