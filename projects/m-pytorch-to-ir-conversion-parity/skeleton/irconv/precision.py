def estimate_ir_size(graph_spec: list[dict], precision: str) -> dict[str, int]:
    raise NotImplementedError


def calculate_fp16_fp32_ratio(graph_spec: list[dict]) -> float:
    raise NotImplementedError
