from vllm_boot.config import MinimalEngineConfig, parse_serve_args


class ServerInstance:
    """Lightweight representation of a running serving engine instance."""

    def __init__(self, config: MinimalEngineConfig):
        raise NotImplementedError

    def boot(self):
        """Initializes and boots the server instance."""
        raise NotImplementedError

    def handle_request(self, path: str) -> dict:
        """Handles incoming HTTP API requests."""
        raise NotImplementedError


def boot_and_query_models(cli_args: list) -> dict:
    """Parses CLI args, boots the server, and queries /v1/models endpoint."""
    raise NotImplementedError
