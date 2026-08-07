from modfilter.matcher import filter_modules


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
