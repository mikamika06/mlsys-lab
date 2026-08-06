def find_first_missing_key(metadata, required_keys):
    for k in required_keys:
        if k not in metadata:
            return k
    return None
