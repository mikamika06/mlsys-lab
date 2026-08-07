def apply_transformer_fusion(model_path: str, output_path: str) -> bool:
    raise NotImplementedError


def get_fused_nodes(model_path: str) -> list:
    raise NotImplementedError


def evaluate_parity(orig_path: str, opt_path: str, inputs: dict) -> float:
    raise NotImplementedError


def measure_phases(orig_path: str, opt_path: str, inputs: dict) -> dict:
    raise NotImplementedError
