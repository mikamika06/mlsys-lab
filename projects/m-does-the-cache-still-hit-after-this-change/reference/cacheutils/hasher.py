import hashlib
import json


def stable_hash(payload):
    def sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in sorted(obj.items())
                    if not str(k).startswith("_volatile")}
        if isinstance(obj, (list, tuple)):
            return [sanitize(x) for x in obj]
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    cleaned = sanitize(payload)
    raw = json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
