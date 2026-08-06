XNNPACK_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "MUL"}
COREML_OPS = {"CONV_2D", "FULLY_CONNECTED", "ADD", "SOFTMAX", "RESHAPE"}

def partition_xnnpack(ops: list[dict]) -> list[dict]:
    """
    Group contiguous XNNPACK-supported ops into DELEGATE ops.
    """
    raise NotImplementedError


def partition_coreml(ops: list[dict]) -> list[dict]:
    """
    Group contiguous CoreML-supported ops into DELEGATE ops.
    """
    raise NotImplementedError
