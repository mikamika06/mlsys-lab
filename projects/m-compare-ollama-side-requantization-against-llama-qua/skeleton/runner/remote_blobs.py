class MockBlobServer:
    """Mock server simulating Ollama blob storage and model creation."""

    def __init__(self):
        raise NotImplementedError

    def upload_blob(self, blob_bytes):
        raise NotImplementedError

    def create_model(self, name, modelfile, blob_digests):
        raise NotImplementedError

    def get_model(self, name):
        raise NotImplementedError


def push_safetensors_as_model(server, model_name, files_map):
    """Upload safetensors file blobs and register remote model."""
    raise NotImplementedError
