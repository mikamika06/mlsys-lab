import hashlib
import hmac


def build_salted_key(tenant_id: str, salt: bytes, prefix_tokens: list) -> str:
    h = hmac.new(salt, digestmod=hashlib.sha256)
    h.update(tenant_id.encode("utf-8"))
    for t in prefix_tokens:
        h.update(int(t).to_bytes(4, byteorder="big", signed=False))
    return h.hexdigest()
