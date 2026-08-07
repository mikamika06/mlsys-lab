def block_hash(tokens: list[int]) -> int:
    raise NotImplementedError


def tokenize_into_blocks(tokens: list[int], block_size: int) -> list[int]:
    raise NotImplementedError


def compute_prefix_match(req_blocks: list[int], worker_blocks: list[int]) -> int:
    raise NotImplementedError


class PrefixRouter:

    def __init__(self, num_workers: int, max_blocks_per_worker: int, block_size: int):
        raise NotImplementedError

    def get_worker_blocks(self, worker_id: int) -> list[int]:
        raise NotImplementedError

    def route(self, tokens: list[int]) -> tuple[int, int]:
        raise NotImplementedError

    def update_cache(self, worker_id: int, tokens: list[int]):
        raise NotImplementedError
