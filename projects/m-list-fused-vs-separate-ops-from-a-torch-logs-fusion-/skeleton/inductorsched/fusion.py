def greedy_pointwise_fuse(nodes):
    """
    Given a list of nodes dicts:
      {'id': str, 'op': str, 'inputs': list[str], 'shape': tuple, 'is_pointwise': bool}
    Return a list of fused kernels (lists of node ids).
    """
    raise NotImplementedError
