def reconstruct_mro(log):
    names = []
    for entry in log:
        prefix, name = entry.split(":", 1)
        if prefix != "super_dispatch":
            raise ValueError("invalid dispatch record")
        names.append(name)
    names.append("object")
    return tuple(names)
