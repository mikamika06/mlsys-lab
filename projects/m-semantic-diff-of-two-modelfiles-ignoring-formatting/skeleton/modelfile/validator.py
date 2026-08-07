VALID_INSTRUCTIONS = {
    "FROM",
    "PARAMETER",
    "SYSTEM",
    "TEMPLATE",
    "ADAPTER",
    "LICENSE",
    "MESSAGE",
}


def validate_modelfile(content: str) -> tuple[bool, int, str]:
    raise NotImplementedError
