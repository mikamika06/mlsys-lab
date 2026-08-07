import re


def parse_scrape(text: str) -> dict:
    families = {}
    types = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split(maxsplit=3)
            if len(parts) >= 4 and parts[1] == "TYPE":
                types[parts[2]] = parts[3]
            continue
        match = re.match(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([^\s]+)(?:\s+\d+)?$",
            line,
        )
        if not match:
            continue
        metric_name, raw_labels, val_str = match.groups()
        try:
            val = float(val_str)
        except ValueError:
            continue
        labels = {}
        if raw_labels:
            for label_pair in re.finditer(
                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"([^"]*)"', raw_labels
            ):
                labels[label_pair.group(1)] = label_pair.group(2)
        family_name = metric_name
        for suffix in ("_bucket", "_count", "_sum", "_created"):
            if metric_name.endswith(suffix):
                base = metric_name[: -len(suffix)]
                if base in types:
                    family_name = base
                    break
        if family_name not in families:
            families[family_name] = {
                "name": family_name,
                "type": types.get(family_name, "untyped"),
                "samples": [],
            }
        families[family_name]["samples"].append(
            {"name": metric_name, "labels": labels, "value": val}
        )
    return families


def compute_p99_ttft(family: dict, target_quantile: float = 0.99) -> dict:
    results = {}
    samples = family.get("samples", [])
    groups = {}
    for s in samples:
        if not s["name"].endswith("_bucket"):
            continue
        labels = s["labels"].copy()
        if "le" not in labels:
            continue
        le_str = labels.pop("le")
        try:
            le_val = float(le_str)
        except ValueError:
            continue
        key = frozenset(labels.items())
        if key not in groups:
            groups[key] = []
        groups[key].append((le_val, s["value"]))
    for key, buckets in groups.items():
        buckets.sort(key=lambda x: x[0])
        if not buckets:
            results[key] = (0.0, 0.0)
            continue
        total_count = buckets[-1][1]
        if total_count <= 0:
            results[key] = (0.0, 0.0)
            continue
        target_count = target_quantile * total_count
        prev_le = 0.0
        prev_count = 0.0
        p99_val = buckets[-1][0]
        err_bound = 0.0
        for le, count in buckets:
            if count >= target_count:
                if count > prev_count:
                    frac = (target_count - prev_count) / (count - prev_count)
                    p99_val = prev_le + frac * (le - prev_le)
                else:
                    p99_val = prev_le
                err_bound = le - prev_le
                break
            prev_le = le
            prev_count = count
        results[key] = (p99_val, err_bound)
    return results


def compute_counter_rates(
    family1: dict, family2: dict, duration_seconds: float
) -> dict:
    results = {}
    if duration_seconds <= 0:
        return results
    s1_map = {
        frozenset(s["labels"].items()): s["value"]
        for s in family1.get("samples", [])
    }
    s2_map = {
        frozenset(s["labels"].items()): s["value"]
        for s in family2.get("samples", [])
    }
    for key, v2 in s2_map.items():
        if key in s1_map:
            v1 = s1_map[key]
            if v2 >= v1:
                delta = v2 - v1
            else:
                delta = v2
            results[key] = delta / duration_seconds
    return results


SCRAPE_1 = """# HELP vllm:time_to_first_token_seconds Time to first token in seconds
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.005",model="llama"} 10.0
vllm:time_to_first_token_seconds_bucket{le="0.01",model="llama"} 40.0
vllm:time_to_first_token_seconds_bucket{le="0.025",model="llama"} 85.0
vllm:time_to_first_token_seconds_bucket{le="0.05",model="llama"} 98.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model="llama"} 100.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model="llama"} 100.0
vllm:time_to_first_token_seconds_sum{model="llama"} 1.42
vllm:time_to_first_token_seconds_count{model="llama"} 100.0
# HELP vllm:num_requests_total Total number of processed requests
# TYPE vllm:num_requests_total counter
vllm:num_requests_total{engine="0",status="finished"} 1500.0
vllm:num_requests_total{engine="1",status="finished"} 1200.0
"""

SCRAPE_2 = """# HELP vllm:time_to_first_token_seconds Time to first token in seconds
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.005",model="llama"} 20.0
vllm:time_to_first_token_seconds_bucket{le="0.01",model="llama"} 70.0
vllm:time_to_first_token_seconds_bucket{le="0.025",model="llama"} 170.0
vllm:time_to_first_token_seconds_bucket{le="0.05",model="llama"} 196.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model="llama"} 200.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model="llama"} 200.0
vllm:time_to_first_token_seconds_sum{model="llama"} 3.10
vllm:time_to_first_token_seconds_count{model="llama"} 200.0
# HELP vllm:num_requests_total Total number of processed requests
# TYPE vllm:num_requests_total counter
vllm:num_requests_total{engine="0",status="finished"} 1750.0
vllm:num_requests_total{engine="1",status="finished"} 50.0
"""

TEST_SCRAPES = [SCRAPE_1, SCRAPE_2]
