import re


def match_target(pattern, name, exact=False):
    if exact:
        return bool(re.fullmatch(pattern, name))
    return bool(re.search(pattern, name))


def compile_targets(patterns, exact=False):
    return [re.compile(p) for p in patterns]
