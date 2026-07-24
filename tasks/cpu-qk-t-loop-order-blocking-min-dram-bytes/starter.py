def qkt_access_order(S: int, d: int, B: int, elem_bytes: int) -> list:
    """
    Return a list of byte addresses in the order accessed by a tile-blocked
    QK^T computation. Q base = 0; K base = S*d*elem_bytes.
    """
    raise NotImplementedError('your code here')
