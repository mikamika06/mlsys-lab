def dump_gguf_json(path: str) -> dict:
    """Dump GGUF metadata and tensor info as JSON-compatible dict."""
    raise NotImplementedError


def patch_metadata(path: str, new_metadata: dict, out_path: str) -> None:
    """Patch metadata in place and preserve tensor bytes."""
    raise NotImplementedError
