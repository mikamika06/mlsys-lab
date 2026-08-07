def aten_op_histogram(exported_program):
    """
    Extracts frequency histogram of core ATen ops from ExportedProgram graph.
    Returns dict mapping str op name (e.g. 'aten.add.Tensor') to int count.
    """
    raise NotImplementedError
