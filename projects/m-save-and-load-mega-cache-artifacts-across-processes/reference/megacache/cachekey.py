import hashlib
import json


def compute_cache_key(config):
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def find_breaking_field(base_config, modified_config):
    all_keys = set(base_config.keys()).union(set(modified_config.keys()))
    for k in sorted(all_keys):
        if base_config.get(k) != modified_config.get(k):
            return k
    return None
