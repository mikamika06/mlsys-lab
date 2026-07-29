from .allocator import Allocator
from . import policy


class Scheduler:
    def __init__(self, config: dict):
        self.cfg = dict(config)
        self.alloc = Allocator(config["num_blocks"], config["block_size"])

    def add(self, requests) -> None:
        raise NotImplementedError

    def step(self) -> dict:
        raise NotImplementedError

    def run(self, max_steps: int = 100000) -> dict:
        raise NotImplementedError
