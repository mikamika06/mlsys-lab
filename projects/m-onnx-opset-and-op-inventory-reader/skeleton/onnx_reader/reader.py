def scan_opsets(model_bytes: bytes) -> dict:
    """
    Parses a binary ONNX model to extract the required opsets.
    Returns a dict mapping domain to version (e.g., {"": 14, "ai.onnx.contrib": 1}).
    ONNX ModelProto opset_import is field 8 (OperatorSetIdProto).
    OperatorSetIdProto has domain (field 1, string, default "") and version (field 2, varint).
    """
    raise NotImplementedError()


def scan_ops(model_bytes: bytes) -> dict:
    """
    Parses a binary ONNX model to extract operator frequencies.
    Returns a dict mapping op_type to count (e.g., {"MatMul": 2, "Relu": 1}).
    ONNX ModelProto graph is field 7 (GraphProto).
    GraphProto node is field 5 (NodeProto).
    NodeProto op_type is field 4 (string).
    """
    raise NotImplementedError()


def estimate_ort_savings(model_bytes: bytes) -> int:
    """
    Calculates the exact byte size of all node `name` (field 3) and `doc_string` (field 6)
    fields inside the graph nodes, including their wire format overhead (tag + length).
    These fields are stripped when converting to .ort format.
    """
    raise NotImplementedError()
