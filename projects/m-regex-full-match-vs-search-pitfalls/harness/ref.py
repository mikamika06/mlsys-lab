import re


def is_matched(pattern, name):
    return bool(re.fullmatch(pattern, name))


def filter_modules(modules, include_patterns, exclude_patterns):
    out = []
    for m in modules:
        included = not include_patterns or any(is_matched(p, m) for p in include_patterns)
        excluded = any(is_matched(p, m) for p in exclude_patterns)
        if included and not excluded:
            out.append(m)
    return out


def apply_rules(modules, rules):
    current = list(modules)
    for rule in rules:
        action = rule.get("action")
        patterns = rule.get("patterns", [])
        if action == "include":
            current = filter_modules(current, patterns, [])
        elif action == "exclude":
            excluded = filter_modules(current, [], patterns)
            current = [m for m in current if m not in excluded]
    return current


TEST_MODULES = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn",
    "model.layers.0.mlp",
    "model.layers.0.mlp.gate_proj",
    "model.norm",
]

INCLUDE_TESTS = [
    (["model.layers.0.mlp"], [], ["model.layers.0.mlp"]),
    (["mlp"], [], []),
    ([], ["model.norm"], ["model.layers.0.self_attn", "model.norm"]),
]
