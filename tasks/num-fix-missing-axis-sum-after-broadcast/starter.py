def broadcast_add(a: list[list[float]], b: list[float]) -> tuple[list[list[float]], callable]:
    """Forward: c = a + b (a: (n,m), b: (m,)). Returns (c, backward).

    Backward takes dc and returns (da, db).  BUGGY — fix the backward pass.
    """
    raise NotImplementedError('your code here')
