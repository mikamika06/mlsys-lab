def diagnose_policy(module_tree, min_size):
    issues = []
    for name, size in module_tree.items():
        if size < min_size:
            issues.append({"module": name, "size": size, "error": "too_small"})
    return issues
