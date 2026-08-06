import math

def sdpa(query: list[list[list[float]]],
         key: list[list[list[float]]],
         value: list[list[list[float]]],
         scale: float | None = None) -> list[list[list[float]]]:
    """
    TODO: This implementation mistakenly applies softmax over the query axis.
    It should be applied over the key dimension (last axis of scores).
    """
    raise NotImplementedError('your code here')
