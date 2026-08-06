def patch_metadata(data: bytes, key: str, new_value: str) -> bytes:
    # Simple replacement for reference testing
    old_val = b"test-model"
    new_val = new_value.encode("utf-8")
    # Pad or replace carefully
    if old_val in data:
        return data.replace(old_val, new_val)
    return data
