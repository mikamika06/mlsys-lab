def fix_schedule(ops):
    names = [name for name, _ in ops]
    deps = {name: list(d) for name, d in ops}
    remaining = set(names)
    done = []
    result = []

    while remaining:
        for name in names:
            if name in remaining and all(dep in done for dep in deps[name]):
                result.append(name)
                done.append(name)
                remaining.remove(name)
                break

    return result
