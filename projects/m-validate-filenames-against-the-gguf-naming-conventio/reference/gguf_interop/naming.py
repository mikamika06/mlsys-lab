import re

def validate_filename(filename):
    pattern = r"^[a-z0-9\-_]+\.(?:q[2-8]_[0-9]|q[2-8]k_[smd]|q[4-5]_[0-9]|q4_k_[sm]|q5_k_[sm]|q6_k|q8_0|f16|f32)\.gguf$"
    match = re.match(pattern, filename.lower())
    if not match:
        return False, {}
    parts = filename.rsplit(".", 2)
    return True, {"model": parts[0], "quant": parts[1]}
