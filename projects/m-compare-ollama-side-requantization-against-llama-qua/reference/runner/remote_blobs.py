import hashlib


def compute_sha256(data):
    """Compute sha256 digest string prefixed with sha256:."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


class MockBlobServer:
    """Mock server simulating Ollama blob storage and model creation."""

    def __init__(self):
        self.blobs = {}
        self.models = {}

    def upload_blob(self, blob_bytes):
        digest = compute_sha256(blob_bytes)
        self.blobs[digest] = blob_bytes
        return {"status": "success", "digest": digest, "size": len(blob_bytes)}

    def create_model(self, name, modelfile, blob_digests):
        for digest in blob_digests:
            if digest not in self.blobs:
                raise ValueError(f"Missing blob digest: {digest}")
        self.models[name] = {
            "modelfile": modelfile,
            "digests": list(blob_digests),
            "total_size": sum(len(self.blobs[d]) for d in blob_digests),
        }
        return {"status": "created", "model": name}

    def get_model(self, name):
        return self.models.get(name)


def push_safetensors_as_model(server, model_name, files_map):
    """Upload safetensors file blobs and register remote model."""
    digests = []
    for _fname, content in files_map.items():
        res = server.upload_blob(content)
        digests.append(res["digest"])
    modelfile = f"FROM {digests[0]}\n" if digests else ""
    return server.create_model(model_name, modelfile, digests)
