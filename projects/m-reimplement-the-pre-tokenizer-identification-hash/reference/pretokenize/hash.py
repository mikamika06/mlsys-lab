import hashlib

def compute_pre_tokenizer_hash(tokenizer_config: dict) -> str:
    chk_txt = tokenizer_config.get("chk_txt", "\n \n\n \n\n\n \t \t\t \t\n \n \n \n \n🚀")
    pre_type = tokenizer_config.get("pre_tokenizer_type", "default")
    payload = f"{pre_type}:{chk_txt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
