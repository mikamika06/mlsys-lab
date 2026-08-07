def parse_header(data: bytes) -> dict:
    """
    Parse a GGUF header from bytes and return a manifest dictionary.
    Manifest should have:
    {
        "magic": str,
        "version": int,
        "tensor_count": int,
        "metadata_kv_count": int,
        "metadata": dict,
        "tensors": list[dict],
        "header_end_offset": int,
        "_meta_end": int
    }
    """
    raise NotImplementedError


def compute_overhead(manifest: dict) -> dict:
    """
    Compute alignment padding waste and container metadata sizes.
    Return dictionary:
    {
        "metadata_bytes": int,
        "tensor_info_bytes": int,
        "padding_waste": int
    }
    """
    raise NotImplementedError
