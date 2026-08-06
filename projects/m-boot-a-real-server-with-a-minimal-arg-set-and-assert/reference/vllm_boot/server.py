from vllm_boot.config import MinimalEngineConfig, parse_serve_args


class ServerInstance:
    """Lightweight representation of a running serving engine instance."""

    def __init__(self, config: MinimalEngineConfig):
        self.config = config
        self.is_running = False

    def boot(self):
        """Initializes and boots the server instance."""
        self.is_running = True

    def handle_request(self, path: str) -> dict:
        """Handles incoming HTTP API requests."""
        if not self.is_running:
            raise RuntimeError("Server is not running")
        if path == "/v1/models":
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.config.served_model_name,
                        "object": "model",
                        "owned_by": "vllm",
                    }
                ],
            }
        raise ValueError(f"Endpoint not found: {path}")


def boot_and_query_models(cli_args: list) -> dict:
    """Parses CLI args, boots the server, and queries /v1/models endpoint."""
    config = parse_serve_args(cli_args)
    server = ServerInstance(config)
    server.boot()
    return server.handle_request("/v1/models")
