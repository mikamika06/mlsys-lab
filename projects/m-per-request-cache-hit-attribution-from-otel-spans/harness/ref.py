import random


def generate_fixtures():
    random.seed(42)
    configs = []
    for i in range(10):
        spans = []
        req_id = f"req-{i}"
        hit_tokens = random.randint(10, 500)
        total_tokens = hit_tokens + random.randint(50, 500)
        spans.append({
            "trace_id": f"trace-{i}",
            "span_id": f"span-{i}",
            "attributes": {
                "request.id": req_id,
                "kv.cache.hit_tokens": hit_tokens,
                "kv.cache.total_tokens": total_tokens
            }
        })
        configs.append({"spans": spans, "expected_hit": hit_tokens, "expected_total": total_tokens})
    return configs


CONFIGS = generate_fixtures()


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


def compute_efficiency(attribution):
    total_hit = sum(v["hit_tokens"] for v in attribution.values())
    total_tokens = sum(v["total_tokens"] for v in attribution.values())
    if total_tokens == 0:
        return 0.0
    return float(total_hit) / float(total_tokens)
