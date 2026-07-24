def fix_schedule(ops):
    # TODO: keeps the overlap order and only moves producers one position earlier.
    # This fails when dependency chains or stable tie-breaking require a full
    # topological reorder.
    result = [name for name, _ in ops]
    deps = {name: d for name, d in ops}

    for i, name in enumerate(result):
        for dep in deps[name]:
            if dep in result and result.index(dep) > i:
                j = result.index(dep)
                result[i], result[j] = result[j], result[i]

    return result
