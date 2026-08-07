import re
from quantlibs.targeting import match_target


def apply_ignore_list(names, ignore_patterns, exact=False):
    out = []
    for name in names:
        ignored = False
        for pat in ignore_patterns:
            if match_target(pat, name, exact=exact):
                ignored = True
                break
        if not ignored:
            out.append(name)
    return out


def filter_modules(module_names, include_patterns, ignore_patterns, exact=False):
    included = []
    for name in module_names:
        matched = False
        for pat in include_patterns:
            if match_target(pat, name, exact=exact):
                matched = True
                break
        if matched:
            included.append(name)
    return apply_ignore_list(included, ignore_patterns, exact=exact)
