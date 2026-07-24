def predict_list_capacities(n: int) -> list[int]:
    result = []
    allocated = 0
    for newsize in range(1, n + 1):
        if newsize > allocated:
            new_allocated = newsize + (newsize >> 3) + 6
            new_allocated &= ~3
            allocated = new_allocated
            result.append(allocated)
    return result
