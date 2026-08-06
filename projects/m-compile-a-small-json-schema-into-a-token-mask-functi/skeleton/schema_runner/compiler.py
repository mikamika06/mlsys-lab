"""JSON Schema to token mask compiler."""

from typing import Any, Callable, Dict, List, Set, Tuple


class SchemaMaskCompiler:
    def __init__(self, vocab: Dict[int, str], eos_token_id: int):
        raise NotImplementedError

    def compile(self, schema: Dict[str, Any]) -> Callable[[List[int]], Set[int]]:
        raise NotImplementedError
