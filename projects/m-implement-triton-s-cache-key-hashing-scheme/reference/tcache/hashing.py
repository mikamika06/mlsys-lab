import hashlib
import json

def compute_cache_key(req):
    canonical = json.dumps(req, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
