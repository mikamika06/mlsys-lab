import numpy as np


class SchemaMasker:
    """Compiles a JSON schema into a state machine token mask."""

    def __init__(self, schema: dict, vocab: list[str]):
        raise NotImplementedError

    def get_mask(self, state: int) -> np.ndarray:
        raise NotImplementedError

    def next_state(self, state: int, token_id: int) -> int:
        raise NotImplementedError

    def is_terminal(self, state: int) -> bool:
        raise NotImplementedError


def compile_schema(schema: dict, vocab: list[str]) -> SchemaMasker:
    """Compiles a small JSON schema into a SchemaMasker instance."""
    raise NotImplementedError
