def get_layer_shapes(config: dict) -> dict[str, tuple[int, int]]:
    raise NotImplementedError


def count_base_params(config: dict) -> int:
    raise NotImplementedError
