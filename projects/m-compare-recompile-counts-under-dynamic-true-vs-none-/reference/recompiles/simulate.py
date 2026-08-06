def simulate_lru_cache(shapes, capacity):
    cache = []
    recompiles = 0
    for shape in shapes:
        if shape in cache:
            cache.remove(shape)
            cache.append(shape)
        else:
            recompiles += 1
            if len(cache) >= capacity:
                cache.pop(0)
            cache.append(shape)
    return recompiles
