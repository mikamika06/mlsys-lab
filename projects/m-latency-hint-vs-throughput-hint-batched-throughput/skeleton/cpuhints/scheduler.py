def derive_config(hint: str, cores: int) -> tuple[int, int]:
    raise NotImplementedError()


def compile_model(model_name: str, cache_dir: str) -> float:
    raise NotImplementedError()


def estimate_throughput(batch_sizes: list[int], hint: str, cores: int) -> dict[int, float]:
    raise NotImplementedError()
