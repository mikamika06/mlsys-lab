import re

def validate_filename(filename: str) -> bool:
    pattern = r"^[a-zA-Z0-9_\-]+-(?:f16|f32|q4_0|q4_1|q5_0|q5_1|q8_0|q4_k_m|q4_k_s)\.gguf$"
    return bool(re.match(pattern, filename))
