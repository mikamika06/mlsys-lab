def rewrite_schema(schema):
    if not isinstance(schema, dict):
        return schema
    res = {}
    for k, v in schema.items():
        if k in ("pattern", "format"):
            continue
        if isinstance(v, dict):
            nv = rewrite_schema(v)
            if nv.get("type") == "string" and "maxLength" not in nv:
                nv["maxLength"] = 100
            res[k] = nv
        elif isinstance(v, list):
            res[k] = [rewrite_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            res[k] = v
    return res
