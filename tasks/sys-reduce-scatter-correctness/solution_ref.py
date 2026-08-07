def reduce_scatter_sum(chunks: list[list[list[float]]]) -> list[list[float]]:
    ranks = len(chunks)
    result = []
    for i in range(ranks):
        chunk_len = len(chunks[0][i])
        total = [0.0] * chunk_len
        for r in range(ranks):
            for k in range(chunk_len):
                total[k] += chunks[r][i][k]
        result.append(total)
    return result
