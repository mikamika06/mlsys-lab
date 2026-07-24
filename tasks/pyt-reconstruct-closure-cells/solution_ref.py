def reconstruct_closure(fn):
    if fn.__closure__ is None:
        return []
    return [
        (name, cell.cell_contents)
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)
    ]
