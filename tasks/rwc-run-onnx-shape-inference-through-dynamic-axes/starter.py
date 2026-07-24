def infer_shapes(input_shapes: dict, graph: list) -> dict:
    """
    Static symbolic shape inference over a small ONNX-style graph.

    input_shapes: dict[str, list] mapping graph-input tensor name -> shape
                  (entries are int or str symbols).
    graph: ordered list of {"name": str, "op": str, "inputs": [str,...],
           "attrs": {...}} nodes; op in {"MatMul","Reshape","Concat","Gather"}.

    Returns a dict mapping every node's name to its inferred output shape
    (list of int/str entries). See task.md for the exact per-op rules.
    """
    raise NotImplementedError('your code here')
