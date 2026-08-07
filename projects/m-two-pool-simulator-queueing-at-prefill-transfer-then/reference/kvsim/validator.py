def validate_kv_transfer_config(config):
    if not isinstance(config, dict):
        return {"valid": False, "error": "not a dictionary"}
    if "kv_connector" not in config:
        return {"valid": False, "error": "missing kv_connector"}
    if "roles" not in config or not isinstance(config["roles"], list):
        return {"valid": False, "error": "missing or invalid roles"}

    ranks = set()
    for item in config["roles"]:
        if not isinstance(item, dict) or "rank" not in item or "role" not in item:
            return {"valid": False, "error": "malformed role entry"}
        r = item["rank"]
        if r in ranks:
            return {"valid": False, "error": "duplicate rank"}
        ranks.add(r)
        if item["role"] not in ("prefill", "decode"):
            return {"valid": False, "error": "invalid role name"}

    if sorted(list(ranks)) != list(range(len(ranks))):
        return {"valid": False, "error": "ranks not contiguous"}

    return {"valid": True, "error": None}
