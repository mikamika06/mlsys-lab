import re


def resolve_by_suffix(named_modules, suffixes):
    matched = []
    for name in sorted(named_modules.keys()):
        for sfx in suffixes:
            if name == sfx or name.endswith("." + sfx):
                matched.append(name)
                break
    return sorted(matched)


def resolve_by_regex(named_modules, pattern):
    compiled = re.compile(pattern)
    matched = [name for name in sorted(named_modules.keys()) if compiled.search(name)]
    return sorted(matched)
