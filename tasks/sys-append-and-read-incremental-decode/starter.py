import math

def incremental_decode(embeddings: list[list[float]], Wq: list[list[float]], Wk: list[list[float]], Wv: list[list[float]]) -> list[list[float]]:
    """Incorrect implementation that only uses the current key/value pair.
This will produce outputs that differ from the correct incremental
attention and therefore fail the grading gate."""
    raise NotImplementedError('your code here')
