class LayerConfig:

    def __init__(self, layer_id: int, is_sliding: bool, window_size: int | None = None):
        raise NotImplementedError


class ModelConfig:

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, layer_configs: list[LayerConfig]):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        raise NotImplementedError
