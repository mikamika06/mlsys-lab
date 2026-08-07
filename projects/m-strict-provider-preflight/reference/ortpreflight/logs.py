import re

PATTERN = re.compile(
    r"(?:Placed\s+node\s+\[?([^\s\]]+)\]?\s+on\s+(?:execution\s+provider|provider)?\s*\[?(\w+)\]?)"
    r"|(?:Node\s+\[?([^\s\]]+)\]?\s+placed\s+on\s+(?:execution\s+provider|provider)?\s*\[?(\w+)\]?)",
    re.IGNORECASE,
)


def parse_ep_node_counts(log_text):
    counts = {}
    for line in log_text.splitlines():
        match = PATTERN.search(line)
        if match:
            groups = match.groups()
            if groups[0] is not None:
                ep = groups[1]
            else:
                ep = groups[3]
            counts[ep] = counts.get(ep, 0) + 1
    return counts


def analyze_ep_distribution(log_text, target_ep):
    counts = parse_ep_node_counts(log_text)
    total_nodes = sum(counts.values())
    target_nodes = counts.get(target_ep, 0)
    fallback_nodes = total_nodes - target_nodes
    fallback_ratio = (
        float(fallback_nodes) / total_nodes if total_nodes > 0 else 0.0
    )
    is_pure = (fallback_nodes == 0) and (total_nodes > 0)
    return {
        "counts": counts,
        "total_nodes": total_nodes,
        "target_nodes": target_nodes,
        "fallback_nodes": fallback_nodes,
        "fallback_ratio": fallback_ratio,
        "is_pure": is_pure,
    }
