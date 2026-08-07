import hashlib
import json


def convert_variant_manifest(raw_manifest):
    def clean_data(obj):
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, dict):
            return {k: clean_data(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return [clean_data(x) for x in obj]
        return obj

    cleaned = clean_data(raw_manifest)
    json_bytes = json.dumps(
        cleaned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest = hashlib.sha256(json_bytes).hexdigest()
    return {"manifest_bytes": json_bytes, "digest": digest}
