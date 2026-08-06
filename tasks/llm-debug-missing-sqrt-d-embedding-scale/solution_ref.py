import math

def normalize_embeddings(embeddings: list[list[float]]) -> list[list[float]]:
    """
    Scale an embedding matrix by 1/sqrt(d), where d is the feature dimension.
    The result is always float64 regardless of input dtype.
    """
    if not embeddings:
        return []
    d = len(embeddings[0])
    scale = 1.0 / math.sqrt(d)

    scaled_matrix = []
    for row in embeddings:
        scaled_row = []
        for val in row:
            scaled_row.append(float(val) * scale)
        scaled_matrix.append(scaled_row)
    return scaled_matrix
