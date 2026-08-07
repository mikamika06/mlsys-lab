from loratarget.matcher import resolve_by_regex, resolve_by_suffix


def compute_param_count(named_modules, module_names):
    total = 0
    for name in module_names:
        if name in named_modules:
            shape = named_modules[name]
            total += shape[0] * shape[1]
    return total


def verify_equivalence(named_modules, pattern, suffixes):
    reg_set = set(resolve_by_regex(named_modules, pattern))
    sfx_set = set(resolve_by_suffix(named_modules, suffixes))
    return reg_set == sfx_set
