"""HuggingFace to MLX tensor remapping logic."""


def match_and_remap_key(hf_key: str, rule_map: dict) -> str | None:
    raise NotImplementedError


def remap_hf_to_mlx(hf_tensors: dict, rule_map: dict) -> tuple[dict, list[str]]:
    raise NotImplementedError
