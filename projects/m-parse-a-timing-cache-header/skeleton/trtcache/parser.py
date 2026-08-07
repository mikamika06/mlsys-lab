def parse_header(data: bytes) -> dict:
    raise NotImplementedError


def is_reusable(cache_header: dict, builder_config: dict) -> bool:
    raise NotImplementedError
