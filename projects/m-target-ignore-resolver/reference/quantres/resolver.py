import fnmatch


def resolve_targets(modules, targets, ignores):
    matched = []
    for m in modules:
        is_target = any(fnmatch.fnmatch(m, t) for t in targets)
        is_ignore = any(fnmatch.fnmatch(m, i) for i in ignores)
        if is_target and not is_ignore:
            matched.append(m)
    return sorted(matched)
