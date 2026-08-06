def resolve_target_modules(named_modules, target_shorthands):
    shorthands = set(target_shorthands)
    resolved = []
    for path in named_modules:
        leaf = path.split(".")[-1]
        if leaf in shorthands or path in shorthands:
            resolved.append(path)
    return sorted(resolved)
