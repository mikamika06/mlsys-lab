def warmup(server, inputs):
    """Warm up server cache using common inputs."""
    for i in inputs:
        server.handle_first_request(str(i))
