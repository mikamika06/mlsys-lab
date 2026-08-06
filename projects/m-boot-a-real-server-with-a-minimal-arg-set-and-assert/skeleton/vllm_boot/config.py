from dataclasses import dataclass


@dataclass
class MinimalEngineConfig:
    model: str
    served_model_name: str
    host: str = "0.0.0.0"
    port: int = 8000


def parse_serve_args(args_list: list) -> MinimalEngineConfig:
    """Parses CLI argument list for serving engine configuration."""
    raise NotImplementedError
