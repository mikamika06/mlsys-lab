def extract_blob(blob: bytes):
    """
    Parse the delegate blob.
    Blob format: b"backend:<name>;ops:<count>;flops:<total>"
    Returns: (backend_name: str, num_ops: int, flops: float)
    """
    raise NotImplementedError


def measure_delegation(partitioned_ops: list[dict]) -> float:
    """
    Returns the fraction of FLOPs that were delegated.
    """
    raise NotImplementedError
