def lookup_embeddings(ids: list[int], weights: list[list[float]]) -> list[list[float]]:
    result = []
    for idx in ids:
        result.append(list(weights[idx]))
    return result
