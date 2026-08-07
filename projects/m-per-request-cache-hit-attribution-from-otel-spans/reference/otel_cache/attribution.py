def extract_attribution(spans):
    res = {}
    for s in spans:
        attrs = s.get("attributes", {})
        rid = attrs.get("request.id")
        if rid:
            res[rid] = {
                "hit_tokens": attrs.get("kv.cache.hit_tokens", 0),
                "total_tokens": attrs.get("kv.cache.total_tokens", 0)
            }
    return res
