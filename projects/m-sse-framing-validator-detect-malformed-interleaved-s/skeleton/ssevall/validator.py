class FramingError(Exception):
    pass


def validate_sse_stream(raw_bytes: bytes) -> list[dict]:
    raise NotImplementedError
