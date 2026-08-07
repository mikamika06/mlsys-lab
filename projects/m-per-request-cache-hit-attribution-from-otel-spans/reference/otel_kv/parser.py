def parse_spans(raw_spans):
    by_req = {}
    for span in raw_spans:
        attrs = span.get("attributes", {})
        req_id = attrs.get("request.id")
        if not req_id:
            continue
        if req_id not in by_req:
            by_req[req_id] = {
                "request_id": req_id,
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "generated_tokens": 0,
                "events": []
            }

        by_req[req_id]["prompt_tokens"] = max(
            by_req[req_id]["prompt_tokens"],
            attrs.get("llm.prompt_tokens", 0)
        )
        by_req[req_id]["cached_tokens"] = max(
            by_req[req_id]["cached_tokens"],
            attrs.get("llm.kv_cache.hit_tokens", 0)
        )
        by_req[req_id]["generated_tokens"] = max(
            by_req[req_id]["generated_tokens"],
            attrs.get("llm.generated_tokens", 0)
        )
        for ev in span.get("events", []):
            if ev.get("name") == "kv_cache_lookup":
                by_req[req_id]["events"].append(ev.get("attributes", {}))

    return list(by_req.values())
