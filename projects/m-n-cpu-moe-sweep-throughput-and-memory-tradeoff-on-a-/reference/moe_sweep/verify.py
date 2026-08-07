def verify_placement(tensor_map, overrides):
    placed = {}
    for name, device in tensor_map.items():
        target = overrides.get(name, device)
        placed[name] = target
    return placed
