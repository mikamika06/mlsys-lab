import ref
from pretokenize.hash import compute_pre_tokenizer_hash

def check(workdir):
    out = {"hashes_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.compute_pre_tokenizer_hash(cfg) if hasattr(ref, "compute_pre_tokenizer_hash") else None
        # Compute reference inline if ref doesn't have it
        import hashlib
        payload = f"{cfg['pre_tokenizer_type']}:{cfg['chk_txt']}".encode("utf-8")
        want_hash = hashlib.sha256(payload).hexdigest()

        got = compute_pre_tokenizer_hash(cfg)
        if got == want_hash:
            ok += 1
    out["hashes_matched"] = float(ok)
    return out
