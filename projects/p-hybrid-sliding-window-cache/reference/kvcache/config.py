class LayerConfig:

    def __init__(self, layer_id: int, is_sliding: bool, window_size: int | None = None):
        if is_sliding and (window_size is None or window_size <= 0):
            raise ValueError("Sliding window layers must specify a positive window_size")
        self.layer_id = layer_id
        self.is_sliding = is_sliding
        self.window_size = window_size


class ModelConfig:

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, layer_configs: list[LayerConfig]):
        if len(layer_configs) != num_layers:
            raise ValueError("Number of layer configs must match num_layers")
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_configs = layer_configs

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        l_configs = []
        for lc in d["layer_configs"]:
            l_configs.append(
                LayerConfig(
                    layer_id=lc["layer_id"],
                    is_sliding=lc["is_sliding"],
                    window_size=lc.get("window_size"),
                )
            )
        return cls(
            num_layers=d["num_layers"],
            num_heads=d["num_heads"],
            head_dim=d["head_dim"],
            layer_configs=l_configs,
        )
