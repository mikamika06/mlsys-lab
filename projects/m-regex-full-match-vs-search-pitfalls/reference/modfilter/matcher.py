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
