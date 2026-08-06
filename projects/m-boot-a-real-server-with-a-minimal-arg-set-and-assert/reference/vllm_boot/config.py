from dataclasses import dataclass


@dataclass
class MinimalEngineConfig:
    model: str
    served_model_name: str
    host: str = "0.0.0.0"
    port: int = 8000


def parse_serve_args(args_list: list) -> MinimalEngineConfig:
    """Parses CLI argument list for serving engine configuration."""
    model = None
    served_model_name = None
    host = "0.0.0.0"
    port = 8000

    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg == "--model" and i + 1 < len(args_list):
            model = args_list[i + 1]
            i += 2
        elif arg == "--served-model-name" and i + 1 < len(args_list):
            served_model_name = args_list[i + 1]
            i += 2
        elif arg == "--host" and i + 1 < len(args_list):
            host = args_list[i + 1]
            i += 2
        elif arg == "--port" and i + 1 < len(args_list):
            port = int(args_list[i + 1])
            i += 2
        else:
            i += 1

    if not model:
        raise ValueError("Missing required parameter: --model")

    if served_model_name is None:
        served_model_name = model

    return MinimalEngineConfig(
        model=model,
        served_model_name=served_model_name,
        host=host,
        port=port,
    )
